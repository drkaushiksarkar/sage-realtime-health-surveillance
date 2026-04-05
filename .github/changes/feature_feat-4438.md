# feat: add retry logic with exponential backoff

## Summary
- Add tenacity-based retry decorator for all external API calls
- Exponential backoff: 1s base, 30s max, 5 max retries
- Circuit breaker after consecutive failures

## Test plan
- [x] Unit test retry behavior with mock failures
- [x] Integration test with actual HTTP timeouts
- [x] Verify
