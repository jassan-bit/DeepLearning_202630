"""Git protocol transport for the scientific GitHub campaign mirror.

The public JSON schema, Merkle inventory, branch isolation and recovery checks
remain in e2_github_campaign. Only transport changes: local Git object plumbing
and two non-force pushes per attempt avoid per-object GitHub REST write quotas.
Credentials are passed only to an ephemeral ASKPASS process environment.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import tempfile
from urllib.parse import unquote, urlsplit

from e2_github_campaign import (CampaignConflict, CampaignRateLimited, CampaignSyncError,
    GitHubCampaignMirror, HEX40, IDENTITY, MAX_ARCHIVE_BYTES, MAX_FILE_BYTES, approved_path)


class GitDataTransport:
    """Implement the mirror's small GitData API using a private bare Git cache."""
    def __init__(self,repository,branch,prefix,*,token=None,_test_remote=None,_allow_local_for_tests=False):
        parts=repository.split("/") if isinstance(repository,str) else []
        if len(parts)!=2 or not all(IDENTITY.fullmatch(p) for p in parts):raise CampaignSyncError("invalid_github_repository")
        if (not isinstance(branch,str) or not branch.startswith("campaigns/")
                or not IDENTITY.fullmatch(branch[10:]) or prefix!=branch):raise CampaignSyncError("campaign_branch_required")
        secret=token if token is not None else os.environ.get("GITHUB_TOKEN")
        if not isinstance(secret,str) or not secret or any(c.isspace() for c in secret):raise CampaignSyncError("github_token_required")
        if _test_remote is not None:
            if not _allow_local_for_tests:raise CampaignSyncError("local_git_remote_is_test_only")
            remote=Path(_test_remote).absolute()
            if not remote.is_dir() or remote.is_symlink():raise CampaignSyncError("invalid_test_git_remote")
            self._remote=str(remote)
        else:self._remote="https://github.com/"+repository+".git"
        self.repository,self.branch,self.prefix=repository,branch,prefix
        self._ref="refs/heads/"+branch
        self._temporary=tempfile.TemporaryDirectory(prefix="e2-scientific-git-")
        self._private=Path(self._temporary.name)
        self._cache=self._private/"objects.git"
        askpass=self._private/"askpass.sh"
        askpass.write_text('#!/bin/sh\ncase "$1" in\n*Username*|*username*) printf "%s\\n" "x-access-token";;\n*Password*|*password*) printf "%s\\n" "$GITHUB_TOKEN";;\n*) exit 1;;\nesac\n')
        askpass.chmod(0o700)
        env=dict(os.environ)
        for key in list(env):
            # Directory, namespace, template and helper overrides can redirect
            # even --git-dir plumbing into an unrelated repository. Preserve
            # only explicit CA trust and restrictive protocol guards, then
            # install our own Git controls below.
            if ((key.startswith("GIT_") and key not in {"GIT_SSL_CAINFO","GIT_SSL_CAPATH",
                    "GIT_ALLOW_PROTOCOL","GIT_PROTOCOL_FROM_USER"})
                    or key in {"SSH_ASKPASS","GH_TOKEN"}):
                env.pop(key,None)
        env.update(GITHUB_TOKEN=secret,GIT_ASKPASS=str(askpass),GIT_TERMINAL_PROMPT="0",
            GIT_CONFIG_GLOBAL=os.devnull,GIT_CONFIG_NOSYSTEM="1",
            LC_ALL="C",
            GIT_AUTHOR_NAME="E2 scientific campaign",GIT_AUTHOR_EMAIL="campaign@users.noreply.github.com",
            GIT_COMMITTER_NAME="E2 scientific campaign",GIT_COMMITTER_EMAIL="campaign@users.noreply.github.com")
        self._environment=env
        self._options=["-c","credential.helper=","-c","core.hooksPath="+os.devnull,
            "-c","http.followRedirects=false","-c","protocol.ext.allow=never","-c","maintenance.auto=false",
            "-c","protocol.file.allow="+("always" if _test_remote is not None else "never")]
        self._known_head=None
        self._pushes=0
        self._maintenance_failed=False
        self._closed=False
        try:
            self._run(["init","--bare","--object-format=sha1",str(self._cache)],bare=False)
            self._run(["remote","add","origin",self._remote])
        except BaseException:
            self.close();raise

    def _run(self,args,*,data=None,environment=None,timeout=60,bare=True,allow_failure=False):
        if self._closed:raise CampaignSyncError("git_transport_closed")
        argv=["git",*self._options]
        if bare:argv += ["--git-dir="+str(self._cache)]
        argv += list(args)
        env=dict(self._environment)
        if environment:env.update(environment)
        process=None
        try:
            process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                     env=env,cwd=self._private,start_new_session=True)
            stdout,stderr=process.communicate(input=data,timeout=timeout)
            result=subprocess.CompletedProcess(argv,process.returncode,stdout,stderr)
        except BaseException as error:
            if process is not None:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                try:process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    if process.stdout is not None:process.stdout.close()
                    if process.stderr is not None:process.stderr.close()
            if isinstance(error,(OSError,subprocess.TimeoutExpired)):
                raise CampaignSyncError("git_request_unconfirmed") from None
            raise
        if result.returncode and not allow_failure:
            error=result.stderr.lower()
            if b"rate limit" in error or b"too many requests" in error or b"429" in error:
                raise CampaignRateLimited()
            if b"non-fast-forward" in error or b"fetch first" in error or b"[rejected]" in error:
                raise CampaignConflict("github_campaign_head_conflict")
            raise CampaignSyncError("git_operation_failed")
        return result

    @staticmethod
    def _sha(value):
        if not isinstance(value,str) or not HEX40.fullmatch(value):raise CampaignSyncError("git_object_invalid")
        return value

    def _object(self,sha,kind):
        self._sha(sha)
        result=self._run(["cat-file","-t",sha])
        if result.stdout.strip()!=kind.encode():raise CampaignSyncError("git_object_type_mismatch")
        return self._run(["cat-file",kind,sha]).stdout

    def _fetch_ref(self):
        listed=self._run(["ls-remote","--exit-code","origin",self._ref],timeout=300,allow_failure=True)
        if listed.returncode==2:
            self._known_head=None
            return 404,{},{}
        if listed.returncode:raise CampaignSyncError("git_remote_read_failed")
        rows=listed.stdout.decode("ascii").splitlines()
        if not rows or any(row.split("\t")[-1]!=self._ref for row in rows):raise CampaignSyncError("git_remote_ref_mismatch")
        self._run(["fetch","--no-tags","--depth=1","origin",self._ref],timeout=300)
        # The branch may advance between listing and fetching; bind the response
        # to the exact fetched commit, which is available in this private cache.
        self._known_head=self._sha(self._run(["rev-parse","FETCH_HEAD"]).stdout.decode().strip())
        return 200,{}, {"object":{"sha":self._known_head}}

    def _archive(self,sha):
        self._object(sha,"commit")
        target=self._private/"snapshot.zip"
        try:
            self._run(["archive","--format=zip","--prefix=snapshot/","--output="+str(target),sha],timeout=300)
            if target.stat().st_size>MAX_ARCHIVE_BYTES:raise CampaignSyncError("git_archive_too_large")
            return 200,{},target.read_bytes()
        finally:target.unlink(missing_ok=True)

    def _write_tree(self,payload):
        entries=payload.get("tree")
        if not isinstance(entries,list):raise CampaignSyncError("git_tree_invalid")
        index=self._private/"index"
        index.unlink(missing_ok=True)
        env={"GIT_INDEX_FILE":str(index)}
        base=payload.get("base_tree")
        self._run(["read-tree",self._sha(base) if base is not None else "--empty"],environment=env)
        lines=[]
        for entry in entries:
            path=entry.get("path")
            if (not isinstance(path,str) or not path.startswith(self.prefix+"/") or "\x00" in path
                    or ".." in PurePosixPath(path).parts or "\\" in path):raise CampaignSyncError("git_tree_path_refused")
            relative=path[len(self.prefix)+1:]
            if relative!="snapshot.json" and not approved_path(relative):raise CampaignSyncError("git_tree_path_refused")
            if entry.get("mode")!="100644" or entry.get("type")!="blob":raise CampaignSyncError("git_tree_entry_refused")
            sha=self._sha(entry.get("sha"))
            lines.append(b"100644 "+sha.encode()+b"\t"+path.encode()+b"\0")
        self._run(["update-index","-z","--index-info"],data=b"".join(lines),environment=env)
        sha=self._sha(self._run(["write-tree"],environment=env).stdout.decode().strip())
        return 201,{}, {"sha":sha}

    def _push(self,sha,*,create):
        sha=self._sha(sha)
        raw=self._object(sha,"commit")
        parents=[line[7:].decode() for line in raw.split(b"\n\n",1)[0].splitlines() if line.startswith(b"parent ")]
        expected=[] if create else [self._known_head]
        if parents!=expected:return 409,{},{}
        # Never use '+', --force or --force-with-lease. A concurrent remote
        # commit makes this sibling commit non-fast-forward and is rejected.
        result=self._run(["push","--porcelain","origin",sha+":"+self._ref],timeout=60,allow_failure=True)
        if result.returncode:
            error=(result.stdout+result.stderr).lower()
            if b"rate limit" in error or b"too many requests" in error or b"429" in error:raise CampaignRateLimited()
            if any(term in error for term in (b"[rejected]",b"non-fast-forward",b"fetch first",b"stale info")):
                return 409,{},{}
            raise CampaignSyncError("git_push_unconfirmed")
        self._known_head=sha
        # Keep every acknowledged commit reachable in this private cache.
        # Maintenance is local, synchronous and follows the remote ACK; it
        # never rewrites the remote branch or changes scientific records.
        self._run(["update-ref","refs/e2-campaign/current",sha])
        self._pushes+=1
        if self._pushes%128==0:
            try:
                maintenance=self._run(["-c","gc.autoDetach=false","gc","--auto"],
                    environment={"GITHUB_TOKEN":""},timeout=300,allow_failure=True)
                self._maintenance_failed=maintenance.returncode!=0
            except CampaignSyncError:
                # The checkpoint is already acknowledged. A local cache
                # maintenance problem must not turn it into an unknown push.
                self._maintenance_failed=True
        return (201 if create else 200),{}, {"object":{"sha":sha}}

    def __call__(self,method,url,headers,payload):
        parsed=urlsplit(url)
        if parsed.scheme!="https" or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise CampaignSyncError("git_transport_url_refused")
        archive_prefix="/"+self.repository+"/zip/"
        if parsed.hostname=="codeload.github.com" and method=="GET" and parsed.path.startswith(archive_prefix):
            return self._archive(self._sha(parsed.path[len(archive_prefix):]))
        api_prefix="/repos/"+self.repository+"/git/"
        if parsed.hostname!="api.github.com" or not parsed.path.startswith(api_prefix):raise CampaignSyncError("git_transport_url_refused")
        route=unquote(parsed.path[len(api_prefix):])
        if method=="GET" and route=="ref/heads/"+self.branch:return self._fetch_ref()
        if method=="GET" and route.startswith("commits/"):
            sha=self._sha(route[8:]);raw=self._object(sha,"commit")
            tree=next((line[5:].decode() for line in raw.splitlines() if line.startswith(b"tree ")),None)
            return 200,{}, {"sha":sha,"tree":{"sha":self._sha(tree)}}
        if method=="GET" and route.startswith("trees/"):
            sha=self._sha(route[6:]);self._object(sha,"tree");entries=[]
            for raw in self._run(["ls-tree","-z",sha]).stdout.split(b"\0"):
                if not raw:continue
                metadata,name=raw.split(b"\t",1);mode,kind,entry_sha=metadata.decode().split()
                entries.append({"path":name.decode("utf-8"),"mode":mode,"type":kind,"sha":entry_sha})
            return 200,{}, {"sha":sha,"tree":entries,"truncated":False}
        if method=="GET" and route.startswith("blobs/"):
            data=self._object(self._sha(route[6:]),"blob")
            if len(data)>MAX_FILE_BYTES:raise CampaignSyncError("git_blob_too_large")
            return 200,{}, {"encoding":"base64","content":base64.b64encode(data).decode(),"size":len(data)}
        if method=="POST" and route=="blobs":
            try:
                if payload.get("encoding")!="base64":raise ValueError()
                data=base64.b64decode(payload["content"],validate=True)
            except (KeyError,ValueError,TypeError):raise CampaignSyncError("git_blob_invalid") from None
            if len(data)>MAX_FILE_BYTES:raise CampaignSyncError("git_blob_too_large")
            sha=self._sha(self._run(["hash-object","-w","--stdin"],data=data).stdout.decode().strip())
            return 201,{}, {"sha":sha}
        if method=="POST" and route=="trees":return self._write_tree(payload)
        if method=="POST" and route=="commits":
            parents=payload.get("parents",[])
            if not isinstance(parents,list) or len(parents)>1:raise CampaignSyncError("git_commit_parents_refused")
            args=["commit-tree",self._sha(payload.get("tree"))]
            for parent in parents:args += ["-p",self._sha(parent)]
            message=payload.get("message")
            if not isinstance(message,str) or len(message)>200:raise CampaignSyncError("git_commit_message_refused")
            sha=self._sha(self._run(args,data=(message+"\n").encode()).stdout.decode().strip())
            return 201,{}, {"sha":sha}
        if method=="POST" and route=="refs" and payload.get("ref")==self._ref:return self._push(payload.get("sha"),create=True)
        if method=="PATCH" and route=="refs/heads/"+self.branch and payload.get("force") is False:
            return self._push(payload.get("sha"),create=False)
        raise CampaignSyncError("git_transport_operation_refused")

    def close(self):
        if not self._closed:
            self._closed=True
            self._environment.pop("GITHUB_TOKEN",None)
            self._temporary.cleanup()


class GitCampaignMirror(GitHubCampaignMirror):
    """The same positive scientific mirror, persisted through Git pushes."""
    def __init__(self,repository,branch,prefix,*,token=None,max_snapshot_bytes=2*1024**3,
                 _test_remote=None,_allow_local_for_tests=False):
        self.git_transport=GitDataTransport(repository,branch,prefix,token=token,
            _test_remote=_test_remote,_allow_local_for_tests=_allow_local_for_tests)
        try:
            super().__init__(repository,branch,prefix,token=token,transport=self.git_transport,max_snapshot_bytes=max_snapshot_bytes)
        except BaseException:
            self.git_transport.close();raise

    def close(self):
        self.git_transport.close()
        self._token=None

    def __enter__(self):return self

    def __exit__(self,*args):self.close();return False
