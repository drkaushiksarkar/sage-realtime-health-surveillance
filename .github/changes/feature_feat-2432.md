# feat: add structured logging with correlation IDs

## Summary
- JSON-formatted logs for production
- Request correlation ID propagation
- Sensitive header redaction
- Configurable log levels per module

## Test plan
- [x] Verify JSON format in production mode
- [x] Verify human-readable format in dev mode
- [x] Test correlation ID propagation across
