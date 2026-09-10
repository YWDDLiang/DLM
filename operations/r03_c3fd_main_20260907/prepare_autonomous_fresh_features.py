"""Cache model-only current views without modifying the historical panels."""
import argparse
from pathlib import Path
from final_improvement import collect_keep_features,write_json


def run(root):
    for name in ('mini_2e6','retained_e3'):
        destination=root/'variants'/name
        bank=destination/'autonomous_keep_fresh'
        if not (bank/'COLLECTION_FINAL.json').exists():
            collect_keep_features(destination,'fresh',bank_override=bank)
    write_json(root/'autonomous_fresh_features/DONE.json',dict(complete=True,
        model_only_current_views=True,physical_results_not_used=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)
