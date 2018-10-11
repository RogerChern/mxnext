### Monkey Patch Technique for Extending Existing Operators

```python
# file: monkey.py

import config
import mxnext as X

old_conv = X.conv

def conv(**kwargs):
    # Do anything you want here according to config
    # Thus config will never contaminate the wrapper
    return old_conv(**kwargs)
    
X.conv = conv
```

```python
# file: start_point.py
import monkey # import monkey patch before import mxnext
import mxnext as X
```