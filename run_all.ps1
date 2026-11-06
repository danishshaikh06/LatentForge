
$attentions = @("mha", "gqa", "mqa", "mla")

foreach ($attention in $attentions) {

    Write-Host "================================="
    Write-Host "Starting $attention"
    Write-Host "================================="

    just train $attention

    if (!$?) {
        throw "Training failed for $attention"
    }

    Write-Host "Finished $attention"
}
