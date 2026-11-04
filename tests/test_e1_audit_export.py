import sys
from pathlib import Path
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from export_e1_audit_improvements import check_file


class AggregateExportTests(unittest.TestCase):
    def check_content(self,name,content):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/name;path.write_text(content)
            check_file(path)

    def test_market_or_prediction_vectors_are_rejected(self):
        for content in ["y_true,y_pred\n1,2\n","date,close\n2025-01-01,100\n"]:
            with self.assertRaises(ValueError):self.check_content("metrics.csv",content)

    def test_weights_and_archives_are_rejected(self):
        for name in ["model.pt","run.tar.gz","market.zip"]:
            with self.assertRaises(ValueError):self.check_content(name,"")

    def test_monthly_predictions_are_allowed_as_aggregates(self):
        self.check_content("monthly_prediction_aggregates.csv","phase,month,n_dates,prediction_mean\ntest,2025-01,31,0.02\n")

    def test_individual_day_disguised_as_month_is_rejected(self):
        with self.assertRaises(ValueError):
            self.check_content("monthly_prediction_aggregates.csv","phase,month,n_dates,prediction_mean\ntest,2025-01-01,1,0.02\n")

    def test_small_residual_bin_must_suppress_statistics(self):
        content="history_days,model,seed,predicted_bin,n_dates,stochastic_training,status,rmse\n7,MLP,4101,1,3,True,suppressed,0.01\n"
        with self.assertRaises(ValueError):self.check_content("residual_bins_by_seed.csv",content)

    def test_suppressed_bin_without_statistics_is_allowed(self):
        content="history_days,model,seed,predicted_bin,n_dates,stochastic_training,status,rmse\n7,MLP,4101,1,3,True,suppressed,\n"
        self.check_content("residual_bins_by_seed.csv",content)


if __name__=="__main__":unittest.main()
