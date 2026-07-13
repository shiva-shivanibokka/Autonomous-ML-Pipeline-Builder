"""
tests.test_sandbox — Tests for E2B sandbox and subprocess execution.
"""


import pandas as pd


class TestSubprocessExecution:
    def test_simple_print(self, tmp_path):
        """Basic Python execution works."""
        from sandbox.executor import _execute_subprocess

        code = "print('test output 123')"
        result = _execute_subprocess(code, str(tmp_path / "dummy.csv"), 30)
        assert result["success"]
        assert "test output 123" in result["stdout"]

    def test_csv_reading(self, tmp_path):
        """Can read a CSV inside subprocess."""
        from sandbox.executor import _execute_subprocess

        csv_path = str(tmp_path / "data.csv")
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        df.to_csv(csv_path, index=False)

        # Use repr() so Windows backslash paths become a valid string literal.
        code = f"""
import pandas as pd
df = pd.read_csv({csv_path!r})
print(f'rows={{len(df)}}')
"""
        result = _execute_subprocess(code, csv_path, 30)
        assert result["success"]
        assert "rows=3" in result["stdout"]

    def test_import_error_detected(self, tmp_path):
        """Import errors are caught and reported."""
        from sandbox.executor import _execute_subprocess

        code = "import nonexistent_package_xyz_abc"
        result = _execute_subprocess(code, str(tmp_path / "dummy.csv"), 30)
        assert result["success"] is False
        assert result["error_text"] or result["stderr"]

    def test_timeout_handling(self, tmp_path):
        """Timeout is enforced."""
        from sandbox.executor import _execute_subprocess

        code = "import time; time.sleep(60)"  # Will time out
        result = _execute_subprocess(code, str(tmp_path / "dummy.csv"), timeout=2)
        assert result["success"] is False
        assert (
            "timeout" in result["error_text"].lower() or "Timeout" in result["stderr"]
        )
