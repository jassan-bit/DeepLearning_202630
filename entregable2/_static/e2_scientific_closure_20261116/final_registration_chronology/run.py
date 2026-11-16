from pathlib import Path
import runpy,sys
r=Path(__file__).resolve().parent
sys.argv=[str(r/'audit_e2_final_registration_chronology.py'),'--verify-bundle',str(r),'--output',str(r/'output')]
runpy.run_path(sys.argv[0],run_name='__main__')
