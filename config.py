SOLVER = 'glpk'
# RNS_FEED_PARAMETERS = {}
RNS_FEED_PARAMETERS = {'source': None,
                       # RNS_FEED_PARAMETERS = {'source': None,
                       'report_diff': False,
                       'on_error': 0}  # 0: quit; 1: report and continue with NRC; -1: silent continue;

INPUT_FILE = {'filename': {'name': 'Input.xlsx'},
              'sheet_feed_lib': {
                  'name': 'Feed Library'},
              'sheet_feeds': {
                  'name': 'Feeds'},
              'sheet_scenario': {
                  'name': 'Scenario'},
              'sheet_batch': {
                  'name': 'Batch'},
              'sheet_lca': {
                  'name': 'LCA'},
              'sheet_lca_lib': {
                  'name': 'LCA Library'},
              'sheet_additives_scenario': {
                  'name': 'Additives'}
              }
