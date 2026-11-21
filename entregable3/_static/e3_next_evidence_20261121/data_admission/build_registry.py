"""Build a dated metadata-only DRAFT; not an acquisition or preregistration."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
from calendar_design import describe

BASE = 'f5c960eefedb5fc49b313dd795b1ec84e784bd78'
ROOT = Path(__file__).resolve().parent


def write(name, value):
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True,
                        help='New empty output directory; never overwrite the frozen proposal')
    args = parser.parse_args()
    ROOT = args.output_root.resolve()
    if ROOT.exists() and any(ROOT.iterdir()):
        raise ValueError('Keep prior outputs; choose a new empty output directory')
    ROOT.mkdir(parents=True, exist_ok=True)
    windows = {'planning_raw_start': '2026-12-01', 'planning_first_target': '2026-12-14',
               'planning_last_target': '2027-02-26', 'confirmation_raw_start': '2027-03-01',
               'confirmation_first_target': '2027-03-15', 'confirmation_last_target': '2028-03-14',
               'actual_planning_freeze_utc': None, 'actual_confirmation_freeze_utc': None,
               'binding_protocol': False, 'final_R': None, 'final_T': None,
               'rule': 'DRAFT dates only. Move before access if scientific readiness or dimensioning requires it; never relabel elapsed/exposed windows as future.'}
    target = {'returns_count': 7, 'required_close_support': 8, 'ddof': 0, 'annualized': False,
              'unit': 'population standard deviation of seven raw log close returns per native observation',
              'primary_metric': 'mean individual-seed RMSE within panel; do not average incomparable panel scales',
              'forecast_steps': 1, 'calendar_duration_equal_across_panels': False}
    identifiers = [('F00','BTCUSDT','Bitcoin spot / USDT','UTC_calendar_daily','Binance Vision','Binance Spot','BTC'),
                   ('F01','SPY','US large-cap equity portfolio ETF','US_regular_equity_sessions','Alpha Vantage','NYSE Arca','shares'),
                   ('F02','USO','Oil futures exposure through exchange-traded fund shares','US_regular_equity_sessions','Alpha Vantage','NYSE Arca','shares'),
                   ('F03','TLT','Long-duration US Treasury bond ETF','US_regular_equity_sessions','Alpha Vantage','Nasdaq','shares')]
    panels = []
    for pid, symbol, domain, clock, provider, venue, units in identifiers:
        crypto = pid == 'F00'
        url = ('https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/1d/' if crypto else
               'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+symbol+'&outputsize=full&apikey={API_KEY}')
        rights = {'terms_URL': ('https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md' if crypto else
                                'https://www.alphavantage.co/terms_of_service/'),
                  'summary': ('CC BY-NC-SA4.0 plus dataset terms v1.0 dated2026-08-26; academic/nonmonetized educational uses listed; owner-specific purpose and future acquisition terms still need review.' if crypto else
                              'Personal non-commercial, private individual access subject to EULA and exceptions; academic institution/public redistribution/external reproducer access are not automatically cleared.'),
                  'specific_use_reviewed_by_owner': False, 'owner_review_record': None,
                  'raw_redistribution_authorized': False, 'global_repository_license_inferred': False}
        axes = [] if crypto else [
            {'axis':'domain','established':True,'evidence':'Issuer metadata verifies '+domain+' rather than BTC crypto exposure; source_observations.json.'},
            {'axis':'size','established':False,'evidence':'Same DRAFT wall-calendar target span gives366 calendar dates versus253 scheduled NYSE sessions; actual complete sizes, native-task equivalence and Nasdaq venue schedule remain unverified.'}]
        panels.append({'id':pid,'instrument':symbol,'domain':domain,'provider':provider,'venue':venue,
                       'source_URL':url,'source_response_version':None,'source_response_bytes':None,'source_response_sha256':None,
                       'provider_symbol_metadata_verified':False,'public_repository_and_external_reproducer_access_reviewed':False,
                       'clock':clock,'clock_timezone':'UTC' if crypto else 'America/New_York',
                       'input_schema':['date','open','high','low','close','volume'],'volume_units':units,
                       'raw_price_policy':'RAW_as_traded_unadjusted',
                       'missing_session_policy':'no_forward_fill_no_synthetic_weekend_bars',
                       'corporate_action_or_roll_policy':('Native spot bars, no futures roll.' if crypto else
                           'Retain raw exchange-traded share prices including dividends/splits under a predeclared rule; USO internal futures rolls are fund economics, not a researcher-spliced futures series. Do not use adjusted close or omit adverse event dates after outcomes.'),
                       'target':deepcopy(target),'draft_windows':deepcopy(windows),'licensing':rights,
                       'guide_difference_axes':axes,'origin_availability_contract_verified':False,
                       'availability_requirement':'Record publication/retrieval/ready timestamps for all past inputs before prediction origin. A later bulk archive or date label alone does not establish contemporaneous availability.',
                       'task_equivalence_review':{'accepted':False,'choice':None,'reviewer':None,'rationale':None,
                           'problem':'A next US session can be separated by1–4 scheduled calendar days; this differs from the original fixed24h UTC guard. No guard change or forward fill is authorized by this proposal.'},
                       'future_venue_calendar':{'URL':'https://www.nyse.com/trade/hours-calendars' if venue=='NYSE Arca' else ('https://www.nasdaqtrader.com/Trader.aspx?id=Calendar' if venue=='Nasdaq' else None),
                           'pinned_and_reviewed':False,'future_Nasdaq_schedule_observed':False if venue=='Nasdaq' else None,
                           'expected_scheduled_confirmation_endpoints':366 if crypto else (253 if venue=='NYSE Arca' else None)},
                       'future_file_metadata':None,'confirmation_endpoint_count_verified':None,
                       'seven_endpoint_boundary_purge_checked':False,'power_dimensioning_accepts_R_and_T':False,
                       'prior_panel_relabelled':False,'admitted_unexposed':False})
    write('concrete_panels.DRAFT.json',{'schema_version':1,'kind':'concrete_unadmitted_prospective_panel_proposals',
        'source_commit':BASE,'prepared_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'DRAFT_NOT_PREREGISTERED_NOT_ADMITTED','current_V3_decision':'NO_GO_CONFIRMATION_CURRENT_RECIPE',
        'original_fixed_calendar_guard_unchanged':True,'acquisition_or_model_execution_enabled':False,
        'R3_accredited':False,'panels':panels,
        'dependence_review':{'pairwise_economic_independence_certified':False,
            'shared_factors':['USD and monetary-policy/global-risk conditions','USO/SPY/TLT share US session clock and provider infrastructure','ETF flows, rates and risk exposures can remain dependent','Providers can share upstream vendor and calendar errors'],
            'ticker_or_provider_changes_not_treated_as_independence':True},
        'scientific_next_action':'Resolve V3/no-GO through development, source/rights and task-equivalence review before admitting and freezing any confirmation.'})
    access = {p['id']:{'outcomes_seen_or_used_for_selection':None,'declaration_before_outcome_access':False,
                       'custodian_access_log_sha256':None,'real_confirmation_freeze_utc':None,
                       'observations':None} for p in panels}
    write('AUTHOR_ACCESS_DECLARATION.template.json',{'schema_version':1,'kind':'blank_real_author_or_custodian_declaration',
        'operator':{'name':None,'role':None,'relationship_to_design':None},'declared_at_utc':None,
        'authentic_external_declaration':False,'signature_claimed':False,'software_identity_authentication':False,
        'panel_access':access})
    write('calendar_design_receipt.json',{'status':'executed_calendar_metadata_arithmetic_only',
        'source_URL':'https://www.nyse.com/trade/hours-calendars','BTC':describe('2027-03-15','2028-03-14','UTC_calendar_daily'),
        'NYSE_Arca':describe('2027-03-15','2028-03-14','NYSE_scheduled_sessions'),
        'Nasdaq_2027_2028_verified':False,'Nasdaq_not_silently_substituted_by_NYSE':True,
        'power_final_R_T_chosen':False,'forecast_clock_equivalence_approved':False})


if __name__ == '__main__':
    main()
