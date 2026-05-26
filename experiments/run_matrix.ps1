# Full experiment matrix: 4 setups x 3 seeds = 12 runs, then evaluate each.
# Usage:  pwsh experiments/run_matrix.ps1 -Episodes 300
param(
    [int]$Episodes = 300,
    [int[]]$Seeds = @(0, 1, 2),
    [int]$EvalEpisodes = 30
)

$env:SDL_VIDEODRIVER = "dummy"

$setups = @(
    @{ Name = "A_window";        Script = "train_dqn_window.py" },
    @{ Name = "B_random";        Script = "train_zdqn_random.py" },
    @{ Name = "C_curl_joint";    Script = "train_curl_joint.py" },
    @{ Name = "D_curl_pretrain"; Script = "train_curl_pretrain.py" }
)

foreach ($setup in $setups) {
    foreach ($seed in $Seeds) {
        Write-Host "=== Training $($setup.Name) seed=$seed ==="
        python $setup.Script --episodes $Episodes --seed $seed
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Training failed for $($setup.Name) seed=$seed" -ForegroundColor Red
            exit $LASTEXITCODE
        }
    }
}

Write-Host "=== Evaluating all runs ==="
foreach ($setup in $setups) {
    foreach ($seed in $Seeds) {
        $dir = "runs/$($setup.Name)_seed$seed"
        python evaluate.py $dir --episodes $EvalEpisodes
    }
}
