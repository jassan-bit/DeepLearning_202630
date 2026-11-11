"""Transport v2: classify explicit HTTP 429 without changing frozen v1 code."""
import re
from e2_github_git_transport import GitDataTransport as _LegacyTransport
from e2_github_git_transport import GitCampaignMirror as _LegacyMirror
from e2_github_campaign import CampaignRateLimited, CampaignSyncError, GitHubCampaignMirror

class GitDataTransport(_LegacyTransport):
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
            # A hexadecimal commit ID can contain 429. Only an HTTP status
            # diagnostic, not an arbitrary substring, establishes that code.
            if (b"rate limit" in error or b"too many requests" in error
                    or re.search(rb"(?:http(?:/\d(?:\.\d)?)?\s+|returned error:\s*)429\b", error)):
                raise CampaignRateLimited()
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


class GitCampaignMirror(_LegacyMirror):
    def __init__(self, repository, branch, prefix, *, token=None,
                 max_snapshot_bytes=2*1024**3, _test_remote=None, _allow_local_for_tests=False):
        self.git_transport = GitDataTransport(repository, branch, prefix, token=token,
            _test_remote=_test_remote, _allow_local_for_tests=_allow_local_for_tests)
        try:
            GitHubCampaignMirror.__init__(self, repository, branch, prefix, token=token,
                transport=self.git_transport, max_snapshot_bytes=max_snapshot_bytes)
        except BaseException:
            self.git_transport.close()
            raise
