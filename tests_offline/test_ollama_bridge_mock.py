# tests_offline/test_ollama_bridge_mock.py
# Pure test of write_modelfile's real Modelfile syntax, no real Ollama
# subprocess or model directory needed.

import tempfile
from pathlib import Path
from scenario_ticket_extraction.ollama_bridge import write_modelfile


def main():
    with tempfile.TemporaryDirectory() as tmp:
        modelfile_path = Path(tmp) / "Modelfile"
        write_modelfile("some/merged/model/dir", str(modelfile_path))
        content = modelfile_path.read_text()
        assert content.startswith("FROM "), f"expected a FROM line first, got: {content!r}"
        assert "merged" in content and "model" in content
        print("  [write_modelfile writes a real FROM line pointing at the merged model dir] OK")

        modelfile_path2 = Path(tmp) / "Modelfile2"
        write_modelfile("some/dir", str(modelfile_path2), system_prompt="Be terse.")
        content2 = modelfile_path2.read_text()
        assert 'SYSTEM """Be terse."""' in content2
        print("  [write_modelfile adds a SYSTEM line when a system_prompt is given] OK")

    print("ollama_bridge offline validation: all checks passed")


if __name__ == "__main__":
    main()
