"""Read-only public-source retrieval for the independent crystal mechanism audit.

Downloads are evidence only; no downloaded source is imported or executed.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import re
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
REPOS = {
    'diffcsppp': ('jiaor17/DiffCSP-PP', 'e82e7ffa7cb2a383bde69b067a343ca137d73e47', [
        'README.md', 'diffcsp/pl_modules/diffusion.py', 'diffcsp/pl_modules/diffusion_w_type.py',
        'diffcsp/pl_modules/cspnet.py', 'diffcsp/pl_modules/lattice/crystal_family.py',
        'diffcsp/pl_modules/lattice/matrix.py', 'diffcsp/common/data_utils.py',
        'conf/train/default.yaml', 'conf/data/mp_20.yaml', 'conf/model/diffusion.yaml',
        'conf/model/decoder/cspnet.yaml', 'scripts/compute_metrics.py', 'scripts/generation.py']),
    'diffcsp': ('jiaor17/DiffCSP', '7121d159826efa2ba9500bf299250d96da37f146', [
        'README.md', 'diffcsp/pl_modules/diffusion.py', 'diffcsp/pl_modules/cspnet.py',
        'conf/train/default.yaml', 'conf/data/mp_20.yaml', 'conf/model/diffusion.yaml']),
    'cdvae': ('txie-93/cdvae', 'f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea', [
        'README.md', 'cdvae/pl_modules/model.py', 'conf/train/default.yaml',
        'conf/data/mp_20.yaml', 'conf/model/vae.yaml', 'scripts/compute_metrics.py']),
    'flowmm': ('facebookresearch/flowmm', '6a96aec3b6eba89f6fa07436f0c8837979abb285', [
        'README.md', 'src/flowmm/model/model_pl.py', 'src/flowmm/model/arch.py',
        'src/flowmm/rfm/manifolds/flat_torus.py', 'src/flowmm/rfm/manifolds/lattice_params.py',
        'src/flowmm/rfm/manifold_getter.py', 'src/flowmm/model/solvers.py',
        'scripts_model/conf/default.yaml', 'scripts_model/conf/data/mp20_llama.yaml',
        'scripts_model/conf/model/null_params.yaml', 'scripts_model/conf/model/abits_params.yaml']),
    'mattergen': ('microsoft/mattergen', '92423660a8bd70e83679086e88f88596d484dc16', [
        'README.md']),
}
PAPERS = {
    'diffcsppp_2402.03992v2': 'https://arxiv.org/html/2402.03992v2',
    'diffcsp_2309.04475v2': 'https://arxiv.org/html/2309.04475v2',
    'cdvae_2110.06197v3': 'https://arxiv.org/html/2110.06197v3',
    'flowmm_2406.04713v1': 'https://arxiv.org/html/2406.04713v1',
    'flowllm_2410.23405v1': 'https://arxiv.org/html/2410.23405v1',
    'mattergen_nature_2025': 'https://www.nature.com/articles/s41586-025-08628-5',
}

def fetch(item):
    name, url, is_html = item
    record = {'path': name, 'url': url, 'accessed_utc': datetime.now(timezone.utc).isoformat()}
    target = ROOT / name
    try:
        response = requests.get(url, timeout=60, headers={'User-Agent': 'Codex-scientific-audit'})
        response.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        record.update(status=response.status_code, bytes=len(response.content), sha256=sha256(response.content).hexdigest())
        if is_html:
            soup = BeautifulSoup(response.content, 'html.parser')
            for tag in soup.find_all(['script', 'style', 'nav']):
                tag.decompose()
            for tag in soup.find_all('math'):
                if tag.get('alttext'):
                    tag.replace_with(' $' + tag.get('alttext') + '$ ')
            body = soup.find('article') or soup
            blocks = []
            for block in body.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table', 'li']):
                if block.find_parent(['table', 'li']):
                    continue
                value = re.sub(r'\s+', ' ', block.get_text(' ', strip=True))
                if value:
                    blocks.append(value)
            target.with_suffix('.txt').write_text('\n\n'.join(blocks), encoding='utf-8')
    except Exception as error:
        record['error'] = str(error)
    return record

if __name__ == '__main__':
    items = []
    for label, (repo, commit, paths) in REPOS.items():
        for path in paths:
            items.append((f'code/{label}/{path}', f'https://raw.githubusercontent.com/{repo}/{commit}/{path}', False))
    items.extend((f'papers/{label}.html', url, True) for label, url in PAPERS.items())
    with ThreadPoolExecutor(max_workers=8) as executor:
        receipts = list(executor.map(fetch, items))
    (ROOT / 'source_receipts.json').write_text(json.dumps(receipts, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'total': len(receipts), 'success': sum('error' not in r for r in receipts), 'failures': [r for r in receipts if 'error' in r]}, ensure_ascii=False))
