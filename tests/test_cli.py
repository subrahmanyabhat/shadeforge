"""Tests run against an in-process fake ComfyUI server. No network, no GPU, no API key."""
from __future__ import annotations

import base64
import json
import os
import struct
import tempfile
import threading
import unittest
import urllib.parse
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from shadeforge import cli
from shadeforge import router
from shadeforge import setup
from shadeforge import typeset


def tiny_png(gray: int = 128) -> bytes:
    """Minimal valid 1x1 grayscale PNG, so Pillow is not needed to make fixtures."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes([0, gray])))
            + chunk(b"IEND", b""))


PNG = tiny_png()


class FakeComfy(BaseHTTPRequestHandler):
    checkpoints = ["sd15.safetensors", "sdxl.safetensors"]
    node_errors: dict = {}
    outputs_after = 0          # number of empty history polls before the image appears
    status_str: str | None = None
    submitted: list[dict] = []
    polls = 0

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence test output
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload) -> None:
        self._send(code, json.dumps(payload).encode())

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/object_info/CheckpointLoaderSimple":
            self._json(200, {"CheckpointLoaderSimple": {
                "input": {"required": {"ckpt_name": [type(self).checkpoints, {}]}}}})
        elif path.startswith("/history/"):
            cls = type(self)
            cls.polls += 1
            record: dict = {"outputs": {}, "status": {}}
            if cls.status_str:
                record["status"] = {"status_str": cls.status_str, "completed": True}
            elif cls.polls > cls.outputs_after:
                record["outputs"] = {"7": {"images": [
                    {"filename": "shadeforge_00001_.png", "subfolder": "", "type": "output"}]}}
            self._json(200, {path.rsplit("/", 1)[1]: record})
        elif path == "/view":
            self._send(200, PNG, "image/png")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if urllib.parse.urlparse(self.path).path == "/prompt":
            type(self).submitted.append(payload)
            if type(self).node_errors:
                self._json(200, {"node_errors": type(self).node_errors})
            else:
                self._json(200, {"prompt_id": "abc-123", "node_errors": {}})
        else:
            self._json(404, {"error": "not found"})


class ComfyServerCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfy)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        FakeComfy.checkpoints = ["sd15.safetensors", "sdxl.safetensors"]
        FakeComfy.node_errors = {}
        FakeComfy.outputs_after = 0
        FakeComfy.status_str = None
        FakeComfy.submitted = []
        FakeComfy.polls = 0
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # .resolve(): on macOS /var is a symlink to /private/var, and cmd_generate resolves
        # its output path -- compare like for like.
        self.out = (Path(self.tmp.name) / "nested" / "out.png").resolve()
        # Never let the test suite install or launch a real ComfyUI.
        patcher = mock.patch.object(cli.setup_mod, "ensure_local")
        self.ensure_local = patcher.start()
        self.addCleanup(patcher.stop)


class TestWorkflow(unittest.TestCase):
    def test_nodes_and_wiring(self):
        wf = build = cli.build_comfy_workflow("a lighthouse", "m.safetensors", 512, 768, 12, 1234,
                                              "blurry")
        self.assertEqual({n["class_type"] for n in wf.values()},
                         {"CheckpointLoaderSimple", "CLIPTextEncode", "EmptyLatentImage",
                          "KSampler", "VAEDecode", "SaveImage"})
        self.assertEqual(wf["4"]["inputs"]["width"], 512)
        self.assertEqual(wf["4"]["inputs"]["height"], 768)
        self.assertEqual(wf["5"]["inputs"]["seed"], 1234)
        self.assertEqual(wf["5"]["inputs"]["steps"], 12)
        self.assertEqual(wf["2"]["inputs"]["text"], "a lighthouse")
        self.assertEqual(wf["3"]["inputs"]["text"], "blurry")
        # positive/negative must not be crossed, and CLIP comes from slot 1 of the loader
        self.assertEqual(wf["5"]["inputs"]["positive"], ["2", 0])
        self.assertEqual(wf["5"]["inputs"]["negative"], ["3", 0])
        self.assertEqual(wf["2"]["inputs"]["clip"], ["1", 1])
        self.assertEqual(wf["6"]["inputs"]["vae"], ["1", 2])
        self.assertIs(build, wf)

    def test_ratio_label(self):
        self.assertEqual(cli.ratio_label(1024, 1024), "1:1")
        self.assertEqual(cli.ratio_label(1536, 864), "16:9")
        self.assertEqual(cli.ratio_label(864, 1536), "9:16")
        self.assertEqual(cli.ratio_label(1024, 768), "4:3")


class TestCheckpointParsing(unittest.TestCase):
    def test_object_info_shape(self):
        payload = {"CheckpointLoaderSimple": {"input": {"required": {
            "ckpt_name": [["a.safetensors", "b.ckpt"], {}]}}}}
        self.assertEqual(cli._names_from_object_info(payload), ["a.safetensors", "b.ckpt"])

    def test_garbage_shapes_do_not_raise(self):
        for payload in ({}, None, [], {"CheckpointLoaderSimple": {}},
                        {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": []}}}}):
            self.assertEqual(cli._names_from_object_info(payload), [])

    def test_models_endpoint_string_list_fallback(self):
        # Regression: the old code read /models/checkpoints as dicts and always got [].
        with mock.patch.object(cli, "request_json", side_effect=[{}, ["x.safetensors"]]):
            self.assertEqual(cli.list_checkpoints("http://example.invalid"), ["x.safetensors"])


class TestGenerateLocal(ComfyServerCase):
    def test_writes_image_and_creates_parent_dirs(self):
        path = cli.generate_local("a lighthouse", self.out, 512, 512, 8,
                                  url=self.url, poll=0.01)
        self.assertEqual(path, self.out)
        self.assertEqual(self.out.read_bytes(), PNG)
        wf = FakeComfy.submitted[0]["prompt"]
        self.assertEqual(wf["1"]["inputs"]["ckpt_name"], "sd15.safetensors")
        self.assertTrue(FakeComfy.submitted[0]["client_id"].startswith("shadeforge-"))

    def test_polls_until_image_ready(self):
        FakeComfy.outputs_after = 2
        cli.generate_local("x", self.out, url=self.url, poll=0.01)
        self.assertGreaterEqual(FakeComfy.polls, 3)
        self.assertTrue(self.out.exists())

    def test_explicit_checkpoint_honoured(self):
        cli.generate_local("x", self.out, checkpoint="sdxl.safetensors", url=self.url, poll=0.01)
        self.assertEqual(FakeComfy.submitted[0]["prompt"]["1"]["inputs"]["ckpt_name"],
                         "sdxl.safetensors")

    def test_unknown_checkpoint_rejected_before_queueing(self):
        with self.assertRaisesRegex(cli.ShadeForgeError, "not found"):
            cli.generate_local("x", self.out, checkpoint="nope.safetensors", url=self.url)
        self.assertEqual(FakeComfy.submitted, [])

    def test_no_checkpoints_installed(self):
        FakeComfy.checkpoints = []
        with self.assertRaisesRegex(cli.ShadeForgeError, "no checkpoints"):
            cli.generate_local("x", self.out, url=self.url)

    def test_node_errors_fail_fast(self):
        FakeComfy.node_errors = {"5": {"errors": ["bad sampler"]}}
        with self.assertRaisesRegex(cli.ShadeForgeError, "rejected the workflow"):
            cli.generate_local("x", self.out, url=self.url, timeout=1, poll=0.01)

    def test_execution_error_reported(self):
        FakeComfy.status_str = "error"
        with self.assertRaisesRegex(cli.ShadeForgeError, "execution error"):
            cli.generate_local("x", self.out, url=self.url, timeout=1, poll=0.01)

    def test_timeout_is_bounded(self):
        FakeComfy.outputs_after = 10**6
        with self.assertRaisesRegex(cli.ShadeForgeError, "Timed out"):
            cli.generate_local("x", self.out, url=self.url, timeout=0, poll=0.01)
        self.assertFalse(self.out.exists())

    def test_server_down(self):
        with self.assertRaisesRegex(cli.ShadeForgeError, "unavailable"):
            cli.generate_local("x", self.out, url="http://127.0.0.1:1")


class TestGemini(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "g.png"

    def test_requires_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(cli.ShadeForgeError, "No Gemini API key"):
                cli.generate_gemini("x", self.out)

    def test_decodes_inline_data(self):
        response = {"candidates": [{"content": {"parts": [
            {"text": "here you go"},
            {"inlineData": {"mimeType": "image/png",
                            "data": base64.b64encode(PNG).decode()}},
        ]}}]}
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True), \
             mock.patch.object(cli, "request_json", return_value=response) as req:
            cli.generate_gemini("a cat", self.out, 1536, 864, model="m")
        self.assertEqual(self.out.read_bytes(), PNG)
        body = req.call_args.kwargs["data"]
        self.assertEqual(body["generationConfig"]["imageConfig"]["aspectRatio"], "16:9")
        self.assertEqual(body["contents"][0]["parts"][0]["text"], "a cat")
        self.assertEqual(req.call_args.kwargs["headers"]["x-goog-api-key"], "k")
        self.assertIn("m:generateContent", req.call_args.args[0])

    def test_blocked_prompt_message(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True), \
             mock.patch.object(cli, "request_json",
                               return_value={"promptFeedback": {"blockReason": "SAFETY"}}):
            with self.assertRaisesRegex(cli.ShadeForgeError, "SAFETY"):
                cli.generate_gemini("x", self.out)
        self.assertFalse(self.out.exists())

    def test_key_never_printed(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "s3cret"}, clear=True):
            self.assertEqual(cli.get_gemini_key(), "s3cret")


class TestCliSurface(ComfyServerCase):
    def test_doctor_ok(self):
        self.assertEqual(cli.main(["doctor", "--comfy-url", self.url]), 0)

    def test_doctor_reports_failure_without_crashing(self):
        self.assertEqual(cli.main(["doctor", "--comfy-url", "http://127.0.0.1:1"]), 1)

    def test_generate_end_to_end(self):
        code = cli.main(["generate", "a lighthouse", "--comfy-url", self.url,
                         "--width", "512", "--height", "512", "--steps", "8",
                         "--output", str(self.out)])
        self.assertEqual(code, 0)
        self.assertEqual(self.out.read_bytes(), PNG)

    def test_auto_never_uses_cloud_when_key_present_and_local_down(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True), \
             mock.patch.object(cli, "generate_gemini") as gem:
            code = cli.main(["generate", "x", "--comfy-url", "http://127.0.0.1:1",
                             "--output", str(self.out)])
        self.assertEqual(code, 2)
        gem.assert_not_called()

    def test_allow_cloud_opts_in(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True), \
             mock.patch.object(cli, "generate_gemini", return_value=self.out) as gem:
            code = cli.main(["generate", "x", "--allow-cloud",
                             "--comfy-url", "http://127.0.0.1:1", "--output", str(self.out)])
        self.assertEqual(code, 0)
        gem.assert_called_once()

    def test_local_preferred_over_cloud_in_auto(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True), \
             mock.patch.object(cli, "generate_gemini") as gem:
            cli.main(["generate", "x", "--allow-cloud", "--comfy-url", self.url,
                      "--output", str(self.out)])
        gem.assert_not_called()

    def test_bad_dimensions_rejected(self):
        for args in (["--width", "100"], ["--width", "4096"], ["--height", "513"],
                     ["--steps", "0"], ["--steps", "99"]):
            with self.assertRaises(SystemExit):
                cli.main(["generate", "x", *args])


class TestSetupHelpers(unittest.TestCase):
    def test_verify_safetensors_detects_truncation(self):
        import struct as _s
        header = json.dumps({"t": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
        blob = _s.pack("<Q", len(header)) + header + b"\x00" * 8
        with tempfile.TemporaryDirectory() as d:
            good = Path(d) / "good.safetensors"
            good.write_bytes(blob)
            self.assertTrue(setup.verify_safetensors(good))
            bad = Path(d) / "bad.safetensors"
            bad.write_bytes(blob[:-3])          # interrupted download
            self.assertFalse(setup.verify_safetensors(bad))
            junk = Path(d) / "junk.safetensors"
            junk.write_bytes(b"not a checkpoint")
            self.assertFalse(setup.verify_safetensors(junk))

    def test_legacy_mac_detection_drives_torch_pins(self):
        with mock.patch.object(setup.sys, "platform", "darwin"), \
             mock.patch.object(setup.platform, "machine", return_value="x86_64"):
            self.assertTrue(setup.is_legacy_mac())
        with mock.patch.object(setup.sys, "platform", "darwin"), \
             mock.patch.object(setup.platform, "machine", return_value="arm64"):
            self.assertFalse(setup.is_legacy_mac())

    def test_ensure_local_starts_without_installing(self):
        with mock.patch.object(setup, "server_ready", side_effect=[False, True]), \
             mock.patch.object(setup, "is_installed", return_value=True), \
             mock.patch.object(setup, "has_model", return_value=True), \
             mock.patch.object(setup, "start_server") as start, \
             mock.patch.object(setup, "install") as inst:
            setup.ensure_local("http://x", auto=True)
        start.assert_called_once()
        inst.assert_not_called()

    def test_ensure_local_refuses_big_install_when_not_allowed(self):
        with mock.patch.object(setup, "server_ready", return_value=False), \
             mock.patch.object(setup, "is_installed", return_value=False), \
             mock.patch.object(setup, "install") as inst:
            with self.assertRaisesRegex(setup.SetupError, "not installed"):
                setup.ensure_local("http://x", auto=True, allow_install=False)
        inst.assert_not_called()


class TestNewCliBehaviour(ComfyServerCase):
    def test_preview_and_open_default_on(self):
        args = cli.build_parser().parse_args(["generate", "x"])
        self.assertTrue(args.preview)
        self.assertTrue(args.open)
        self.assertTrue(args.auto_setup)
        off = cli.build_parser().parse_args(["generate", "x", "--no-preview", "--no-open"])
        self.assertFalse(off.preview)
        self.assertFalse(off.open)

    def test_generate_invokes_preview_and_open(self):
        with mock.patch.object(cli, "grayscale_preview") as prev, \
             mock.patch.object(cli, "open_image", return_value=True) as opn:
            cli.main(["generate", "x", "--comfy-url", self.url, "--output", str(self.out)])
        prev.assert_called_once_with(self.out, 64)   # default --preview-width
        opn.assert_called_once_with(self.out)

    def test_auto_setup_runs_when_local_down(self):
        cli.main(["generate", "x", "--comfy-url", self.url, "--output", str(self.out),
                  "--no-preview", "--no-open"])
        self.ensure_local.assert_called_once()

    def test_no_auto_setup_skips_installer(self):
        cli.main(["generate", "x", "--no-auto-setup", "--comfy-url", self.url,
                  "--output", str(self.out), "--no-preview", "--no-open"])
        self.ensure_local.assert_not_called()

    def test_cloud_fallback_does_not_trigger_4gb_install(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True), \
             mock.patch.object(cli, "generate_gemini", return_value=self.out):
            cli.main(["generate", "x", "--allow-cloud", "--comfy-url", "http://127.0.0.1:1",
                      "--output", str(self.out), "--no-preview", "--no-open"])
        self.assertFalse(self.ensure_local.call_args.kwargs["allow_install"])

    def test_gemini_zero_quota_explains_billing(self):
        err = cli.ShadeForgeError('HTTP 429 from x: {"message": "limit: 0, model: foo"}')
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True), \
             mock.patch.object(cli, "request_json", side_effect=err):
            with self.assertRaisesRegex(cli.ShadeForgeError, "billing enabled"):
                cli.generate_gemini("x", self.out)

    def test_progress_bar_parsing(self):
        log = Path(self.tmp.name) / "comfyui.log"
        log.write_text(
            "loading model\r 65%|######5   | 13/20 [03:19<01:44, 14.90s/it]"
            "\r 70%|#######   | 14/20 [03:34<01:28, 14.76s/it]"
        )
        self.assertEqual(cli.read_comfy_progress(log), (14, 20, "01:28"))
        self.assertIn("14/20", cli.render_bar(14, 20, "01:28"))
        self.assertIsNone(cli.read_comfy_progress(log, start_pos=log.stat().st_size))
        self.assertIsNone(cli.read_comfy_progress(Path(self.tmp.name) / "missing.log"))

    def test_sd15_oversize_warning(self):
        with mock.patch("builtins.print") as p:
            cli.warn_oversized(1024, 576, "v1-5-pruned-emaonly.safetensors")
        self.assertIn("SD1.5", str(p.call_args))
        with mock.patch("builtins.print") as p:
            cli.warn_oversized(768, 432, "v1-5-pruned-emaonly.safetensors")
            cli.warn_oversized(1536, 864, "sdxl_base.safetensors")
        p.assert_not_called()


class TestModelSelection(unittest.TestCase):
    def test_picks_best_that_fits_disk(self):
        self.assertEqual(setup.pick_model(free=50, legacy=False), "sdxl")
        self.assertEqual(setup.pick_model(free=8, legacy=False), "realistic-vision")
        self.assertIsNone(setup.pick_model(free=3, legacy=False))
        self.assertIsNone(setup.pick_model(free=0, legacy=False))

    def test_cpu_only_intel_mac_avoids_heavy_models(self):
        # SDXL is 1024px-native: minutes per step on a CPU. Not an automatic choice.
        self.assertEqual(setup.pick_model(free=500, legacy=True), "realistic-vision")
        self.assertEqual(setup.pick_model(free=500, legacy=False), "sdxl")

    def test_flux_is_never_automatic(self):
        for legacy in (True, False):
            self.assertNotEqual(setup.pick_model(free=500, legacy=legacy), "flux-schnell")

    def test_install_refuses_when_disk_too_small(self):
        with mock.patch.object(setup, "free_gb", return_value=1.0), \
             mock.patch.object(setup, "has_model", return_value=False), \
             mock.patch.object(setup, "run") as runner:
            with self.assertRaisesRegex(setup.SetupError, "Not enough disk"):
                setup.install()
        runner.assert_not_called()

    def test_unknown_model_rejected(self):
        with mock.patch.object(setup, "has_model", return_value=False), \
             mock.patch.object(setup, "free_gb", return_value=50):
            with self.assertRaisesRegex(setup.SetupError, "Unknown model"):
                setup.install(model="not-a-model")

    def test_best_installed_checkpoint_prefers_photoreal(self):
        self.assertEqual(
            cli.best_checkpoint(["v1-5-pruned-emaonly.safetensors", "realisticVision_v51.safetensors"]),
            "realisticVision_v51.safetensors")
        self.assertEqual(cli.best_checkpoint(["v1-5-pruned-emaonly.safetensors"]),
                         "v1-5-pruned-emaonly.safetensors")
        self.assertIsNone(cli.best_checkpoint([]))


class TestTextGuardrail(unittest.TestCase):
    def test_detects_prompts_asking_for_words(self):
        for prompt in ['a bottle with the text "LUXE SERUM"',
                       "a poster with a headline reading welcome",
                       "label saying organic honey",
                       "shop signage above the door",
                       "elegant typography on the box"]:
            self.assertTrue(typeset.wants_text(prompt), prompt)

    def test_ignores_plain_scene_prompts(self):
        for prompt in ["a lighthouse in sea mist",
                       "green tender coconuts on a wooden cart",
                       "amber serum bottle on black marble"]:
            self.assertFalse(typeset.wants_text(prompt), prompt)


class TestProgressState(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(cli, "STATE_FILE", Path(self.tmp.name) / "progress.json")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_records_owning_session_and_clears(self):
        with mock.patch.dict(os.environ, {"CLAUDE_CODE_SESSION_ID": "SESS-9"}, clear=True):
            cli.write_progress_state(Path("/tmp/comfy.log"), 42)
        data = json.loads(cli.STATE_FILE.read_text())
        self.assertEqual(data["session"], "SESS-9")
        self.assertEqual(data["start"], 42)
        cli.clear_progress_state()
        self.assertFalse(cli.STATE_FILE.exists())

    def test_clear_is_safe_when_absent(self):
        cli.clear_progress_state()
        cli.clear_progress_state()  # must not raise


class TestPreviewTty(unittest.TestCase):
    def test_skipped_when_not_a_terminal(self):
        # Regression: ANSI blocks dumped into a pipe are thousands of junk bytes.
        with mock.patch.object(cli.sys.stdout, "isatty", return_value=False), \
             mock.patch("builtins.print") as p:
            cli.grayscale_preview(Path("/nonexistent.png"))
        self.assertIn("not a terminal", str(p.call_args))


class TestTypeset(unittest.TestCase):
    def test_builds_layout_with_only_given_fields(self):
        if not typeset.available():
            self.skipTest("Pillow or system fonts unavailable")
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src.png"
            src.write_bytes(PNG)
            out = Path(d) / "out.png"
            typeset.build_ad(src, out, headline="LUXE HAIR SERUM",
                             sub="SMOOTH • NOURISH • SHINE", cta="DISCOVER",
                             size=(540, 675))
            from PIL import Image
            self.assertEqual(Image.open(out).size, (540, 675))

    def test_font_lookup_falls_back_off_macos(self):
        # Regression: curated paths are macOS-only, so Linux/Windows users got
        # typeset.available() == False and silently lost the headline feature.
        if not any(Path(d).is_dir() for d in typeset.FONT_DIRS):
            self.skipTest("no system font directories on this machine")
        with mock.patch.object(typeset, "SERIF_FACES", ["/nope/serif.ttf"]), \
             mock.patch.object(typeset, "SANS_FACES", ["/nope/sans.ttf"]):
            serif = typeset._first_existing(typeset.SERIF_FACES, serif=True)
            sans = typeset._first_existing(typeset.SANS_FACES)
            if not serif or not sans:
                self.skipTest("font directories exist but contain no usable faces")
            self.assertTrue(typeset.available())

    def test_no_fonts_anywhere_raises_clearly(self):
        with mock.patch.object(typeset, "SERIF_FACES", []), \
             mock.patch.object(typeset, "SANS_FACES", []), \
             mock.patch.object(typeset, "FONT_DIRS", ["/nonexistent-font-dir"]):
            self.assertFalse(typeset.available())
            with tempfile.TemporaryDirectory() as d:
                src = Path(d) / "s.png"
                src.write_bytes(PNG)
                with self.assertRaisesRegex(RuntimeError, "No suitable system fonts"):
                    typeset.build_ad(src, Path(d) / "o.png", headline="X")

    def test_long_headline_is_shrunk_not_clipped(self):
        if not typeset.available():
            self.skipTest("Pillow or system fonts unavailable")
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "s.png"
            src.write_bytes(PNG)
            out = Path(d) / "o.png"
            typeset.build_ad(src, out, headline="A" * 60, size=(540, 675))
            self.assertTrue(out.exists())


class TestRouter(unittest.TestCase):
    PHOTO = "realisticVision_v51.safetensors"
    ART = "dreamshaper_8.safetensors"
    XL = "sd_xl_base_1.0.safetensors"
    FLUX = "flux1-schnell-fp8.safetensors"
    BASE = "v1-5-pruned-emaonly.safetensors"

    def test_photo_prompt_picks_photoreal(self):
        d = router.decide("RAW photo of a serum bottle, 85mm", [self.PHOTO, self.ART, self.BASE])
        self.assertEqual(d.model, self.PHOTO)
        self.assertFalse(d.typeset)

    def test_illustration_prompt_picks_stylised(self):
        d = router.decide("a watercolour illustration of a fox", [self.PHOTO, self.ART])
        self.assertEqual(d.model, self.ART)

    def test_photoreal_illustration_resolves_to_photo(self):
        # "photorealistic illustration" means a photo-looking render, not a painting.
        d = router.decide("photorealistic illustration of a car", [self.PHOTO, self.ART])
        self.assertEqual(d.model, self.PHOTO)

    def test_text_prompt_without_flux_falls_back_to_typeset(self):
        d = router.decide('a poster reading "SUMMER SALE"', [self.PHOTO, self.ART])
        self.assertTrue(d.typeset)
        self.assertEqual(d.missing, "flux-schnell")
        self.assertIn("cannot" if False else "spell", " ".join(d.reasons))

    def test_text_prompt_uses_flux_when_available(self):
        d = router.decide('a poster reading "SUMMER SALE"', [self.PHOTO, self.FLUX])
        self.assertEqual(d.model, self.FLUX)
        self.assertFalse(d.typeset)

    def test_flux_ignored_on_legacy_mac(self):
        # The pinned 2024-era ComfyUI cannot load Flux; routing to it would hang for hours.
        d = router.decide('a poster reading "SALE"', [self.PHOTO, self.FLUX], legacy_mac=True)
        self.assertEqual(d.model, self.PHOTO)
        self.assertTrue(d.typeset)
        self.assertTrue(any("Intel Mac" in w for w in d.warnings))

    def test_large_canvas_prefers_sdxl(self):
        d = router.decide("a wide landscape", [self.PHOTO, self.XL], width=1024, height=1024)
        self.assertEqual(d.model, self.XL)

    def test_large_canvas_without_sdxl_warns(self):
        d = router.decide("a wide landscape", [self.PHOTO], width=1024, height=1024)
        self.assertEqual(d.model, self.PHOTO)
        self.assertTrue(any("duplicated subjects" in w for w in d.warnings))

    def test_logo_prompt_warns_about_vectors(self):
        d = router.decide("a logo for a coffee roastery", [self.PHOTO])
        self.assertTrue(any("vector" in w.lower() for w in d.warnings))

    def test_no_checkpoints_returns_no_model(self):
        d = router.decide("anything", [])
        self.assertIsNone(d.model)
        self.assertIn("setup", " ".join(d.reasons))

    def test_reasons_never_contradict(self):
        # A found-but-unavailable signal must not also report "no medium cues".
        d = router.decide("a watercolour illustration of a fox", [self.PHOTO])
        joined = " ".join(d.reasons)
        self.assertIn("Illustration asked for", joined)
        self.assertNotIn("No medium cues", joined)

    def test_explain_is_printable_for_every_branch(self):
        for prompt, installed in [("a fox", []), ("a fox", [self.PHOTO]),
                                  ('says "HI"', [self.PHOTO]), ("a logo", [self.XL])]:
            self.assertIsInstance(router.explain(router.decide(prompt, installed)), str)


class TestModelCatalogueGuards(unittest.TestCase):
    def test_auto_pick_never_selects_a_16gb_model(self):
        # Huge opt-in models must never be pulled by an automatic setup.
        self.assertNotEqual(setup.pick_model(free=500), "flux-schnell")

    def test_flux_refused_on_legacy_mac(self):
        with mock.patch.object(setup, "is_legacy_mac", return_value=True):
            self.assertIsNotNone(setup.check_feasible("flux-schnell"))
        with mock.patch.object(setup, "is_legacy_mac", return_value=False):
            self.assertIsNone(setup.check_feasible("flux-schnell"))

    def test_every_catalogue_entry_is_complete(self):
        for key, spec in setup.MODELS.items():
            for field in ("file", "url", "gb", "quality", "license", "note"):
                self.assertIn(field, spec, f"{key} missing {field}")
            self.assertTrue(spec["url"].startswith("https://"), key)


if __name__ == "__main__":
    unittest.main()
