$target = 906
$py = 'D:\anaconda3\envs\simpleclick-dinov3\python.exe'
$workdir = 'D:\code\论文代码\simpleclick+dinov3'
$evalDir = Join-Path $workdir 'outputs\head_oracle_eval'
$runnerLog = Join-Path $evalDir 'runner_autoresume.log'
$runnerErr = Join-Path $evalDir 'runner_autoresume.err.log'
$monitorLog = Join-Path $evalDir 'monitor.log'

Set-Location $workdir

$progressCode = @'
import json, glob, os
from pathlib import Path
files = glob.glob('outputs/head_oracle_eval/*.json')
seen = {}
for p in files:
    name = os.path.basename(p)
    if name.endswith('.samples.json') or name == 'skipped_keys.json':
        continue
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        continue
    if not isinstance(data, dict):
        continue
    for item in data.get('samples', []):
        key = item.get('sample_key') or '{}:{}:{}'.format(item.get('sample_id'), item.get('object_id'), item.get('part_id'))
        seen[key] = item
skip = []
skip_path = Path('outputs/head_oracle_eval/skipped_keys.json')
if skip_path.exists():
    with open(skip_path, 'r', encoding='utf-8') as f:
        skip = json.load(f)
print(json.dumps({'done': len(seen), 'skipped': len(skip)}, ensure_ascii=False))
'@

while ($true) {
    $status = & $py -c $progressCode | ConvertFrom-Json
    $done = [int]$status.done
    $skipped = [int]$status.skipped
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -like 'python*' -and $_.CommandLine -match 'head_oracle_eval|run_part_noc85_resilient|oracle-granularity'
    }
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path $monitorLog -Value "[$ts] done=$done skipped=$skipped active=$($procs.Count)"
    if (($done + $skipped) -ge $target -and $procs.Count -eq 0) {
        break
    }
    if ($procs.Count -eq 0 -and ($done + $skipped) -lt $target) {
        Add-Content -Path $monitorLog -Value "[$ts] resuming oracle head evaluation from done=$done skipped=$skipped"
        Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList '-u','-m','granularity_recovery.run_part_noc85_resilient','--base-granularity-checkpoint','experiments/granularity_refine_rerun/best.pth','--part-name','head','--chunk-size','5','--max-clicks','20','--oracle-granularity','--output-dir','outputs/head_oracle_eval','--seed-result-globs','__none__','--seed-skip-sample-ids','__none__' -WorkingDirectory $workdir -RedirectStandardOutput $runnerLog -RedirectStandardError $runnerErr | Out-Null
        Start-Sleep -Seconds 5
    }
    Start-Sleep -Seconds 30
}
