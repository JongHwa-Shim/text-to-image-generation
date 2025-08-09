# 분산학습용 네트워크 속도 테스트 스크립트
param(
    [string[]]$WorkerIPs = @(163.152.176.12),
    [string]$TestDuration = "30"
)

Write-Host "🌐 분산학습 네트워크 속도 테스트 시작" -ForegroundColor Green
Write-Host "=" * 50

# 1. 로컬 네트워크 어댑터 정보
Write-Host "📊 로컬 네트워크 어댑터 정보:" -ForegroundColor Yellow
Get-NetAdapter | Where-Object {$_.Status -eq "Up"} | Select-Object Name, InterfaceDescription, LinkSpeed | Format-Table -AutoSize

# 2. 워커 노드들과의 연결 테스트
if ($WorkerIPs.Count -gt 0) {
    Write-Host "🔗 워커 노드 연결 테스트:" -ForegroundColor Yellow
    
    foreach ($ip in $WorkerIPs) {
        Write-Host "테스트 중: $ip" -ForegroundColor Cyan
        
        # Ping 테스트
        $pingResult = Test-NetConnection -ComputerName $ip -InformationLevel Quiet
        if ($pingResult) {
            $ping = ping $ip -n 4 | Select-String "평균"
            Write-Host "  ✅ Ping: $ping" -ForegroundColor Green
            
            # 포트 테스트 (분산학습 포트)
            $portTest = Test-NetConnection -ComputerName $ip -Port 29500 -InformationLevel Quiet
            if ($portTest) {
                Write-Host "  ✅ 포트 29500: 열림" -ForegroundColor Green
            } else {
                Write-Host "  ❌ 포트 29500: 닫힘" -ForegroundColor Red
            }
        } else {
            Write-Host "  ❌ 연결 실패" -ForegroundColor Red
        }
    }
}

# 3. 인터넷 속도 테스트 (참고용)
Write-Host "🌍 인터넷 속도 테스트 (참고용):" -ForegroundColor Yellow
Write-Host "온라인 속도 테스트를 위해 다음 사이트를 방문하세요:"
Write-Host "- https://www.speedtest.net"
Write-Host "- https://fast.com"

# 4. 실시간 네트워크 사용량 모니터링
Write-Host "📈 실시간 네트워크 사용량 ($TestDuration초):" -ForegroundColor Yellow
$counter = 0
while ($counter -lt $TestDuration) {
    try {
        $networkStats = Get-Counter "\Network Interface(*)\Bytes Total/sec" -MaxSamples 1 -ErrorAction SilentlyContinue
        $totalBytes = ($networkStats.CounterSamples | Where-Object {$_.InstanceName -notlike "*isatap*" -and $_.InstanceName -ne "_Total"} | Measure-Object -Property CookedValue -Sum).Sum
        $mbps = ($totalBytes * 8) / 1MB
        Write-Host "현재 사용량: $([math]::Round($mbps, 2)) Mbps" -ForegroundColor Cyan
    } catch {
        Write-Host "네트워크 모니터링 오류" -ForegroundColor Red
    }
    Start-Sleep 1
    $counter++
}

# 5. 권장사항 출력
Write-Host "`n💡 분산학습 네트워크 권장사항:" -ForegroundColor Yellow
Write-Host "- 최소 요구: 10Gbps 이상"
Write-Host "- 권장: 25Gbps 이상"
Write-Host "- 최적: 100Gbps 이상 (InfiniBand)"
Write-Host "- 지연시간: 1ms 이하 권장"

Write-Host "`n✅ 네트워크 테스트 완료" -ForegroundColor Green
