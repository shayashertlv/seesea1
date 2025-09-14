# seesea1

## Logging

This project uses Python's built-in `logging` module instead of `print` statements.
Each module defines a logger with `logger = logging.getLogger(__name__)` so it
honors the logging configuration of the calling application.

### Enable log messages
Add the following early in your application to see INFO and WARNING messages:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Silence log messages
Raise the global log level or adjust individual module levels:

```python
import logging
logging.getLogger('seg_mask2former').setLevel(logging.ERROR)
logging.getLogger('seg_sam2').setLevel(logging.ERROR)
logging.getLogger('sr_corps').setLevel(logging.ERROR)
```

Setting the level to `ERROR` suppresses the informational warnings issued by
these modules.

