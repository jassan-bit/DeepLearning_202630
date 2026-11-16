from pathlib import Path
import runpy,sys
root=Path(__file__).resolve().parent
sys.argv=[str(root/'audit_e2_registration_chronology.py'),'--verify-bundle',str(root),'--output',str(root/'output')]
runpy.run_path(sys.argv[0],run_name='__main__')
