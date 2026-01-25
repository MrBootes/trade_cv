from __future__ import annotations

import atexit
import base64
import html
import json
import re
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots



pio.templates.default = "plotly_white"


RESULT_FIELDS = (
 "tickers_profit",
 "lookup_days",
 "tickers",
 "types",
 "ticker_volume_prices",
 "cash_profit_array",
 "total_profit",
 "total_cash",
 "total_credit",
 "total_value",
 "types_unique",
 "types_profits",
 "types_volume_prices",
 "start_money",
)


_FREQ_LABELS: Tuple[str, str, str] = ("Daily", "Weekly", "Monthly")
_FREQ_RESULT_INDICES = {0, 1, 4, 5, 6, 7, 8, 9, 11, 12}


def _has_frequency_bundle(results: Any) -> bool:
	if not isinstance(results, (list, tuple)):
		return False
	for i in _FREQ_RESULT_INDICES:
		if i >= len(results):
			continue
		v = results[i]
		if isinstance(v, (list, tuple)) and len(v) == 3:
			return True
	return False


def _results_view(results: List[Any], freq_index: int) -> List[Any]:
	fi = int(max(0, min(2, freq_index)))
	out = list(results)
	for i in _FREQ_RESULT_INDICES:
		if i >= len(out):
			continue
		v = out[i]
		if isinstance(v, (list, tuple)) and len(v) == 3:
			out[i] = v[fi]
	return out


def _title_with_frequency(title: Any, freq_label: str) -> str:
	base = str(title) if title is not None else ""
	base = base.strip()
	for fl in _FREQ_LABELS:
		base = re.sub(r"\s*\(" + re.escape(fl) + r"\)\s*$", "", base)
	if not base:
		return f"{freq_label}"
	return f"{base} ({freq_label})"


def _trace_visible_value(tr: Any) -> Any:
	try:
		v = tr.visible
		return True if v is None else v
	except Exception:
		return True


def _build_frequency_switched_figure(
	raw_builder: Callable[[List[Any], Optional[str]], go.Figure],
	results: List[Any],
	board: Optional[str],
) -> go.Figure:
	if not _has_frequency_bundle(results):
		return raw_builder(results, board)

	figs: List[go.Figure] = []
	for fi in range(3):
		out = raw_builder(_results_view(results, fi), board)
		if not isinstance(out, go.Figure):
			raise TypeError(f"Expected go.Figure from builder, got {type(out)}")
		figs.append(out)

	def _layout_meta_to_dict(fig: go.Figure) -> Dict[str, Any]:
		try:
			m = getattr(fig.layout, "meta", None)
		except Exception:
			m = None
		out: Dict[str, Any] = {}
		if m is None:
			return out
		try:
			if isinstance(m, dict):
				out.update(m)
			elif hasattr(m, "to_plotly_json"):
				mm = m.to_plotly_json()
				if isinstance(mm, dict):
					out.update(mm)
			else:
				out.update(dict(m))
		except Exception:
			return {}
		return {k: v for k, v in out.items() if isinstance(k, str) and k.startswith("tc_") and k != "tc_freq_switch"}

	n = len(figs[0].data)
	if any(len(f.data) != n for f in figs):
		return figs[0]

	base = figs[0]
	for fi in (1, 2):
		base.add_traces(figs[fi].data)

	for idx in range(n, 3 * n):
		try:
			base.data[idx].visible = False
		except Exception:
			pass

	vis_states = []
	for fi in range(3):
		vis_states.append([_trace_visible_value(tr) for tr in figs[fi].data])
	views_meta = []
	for fi, label in enumerate(_FREQ_LABELS):
		visible: List[Any] = [False] * (3 * n)
		for j in range(n):
			visible[fi * n + j] = vis_states[fi][j]
		views_meta.append(
			dict(
				label=label,
				visible=visible,
				relayout={},
			)
		)

	try:
		existing_meta = getattr(base.layout, "meta", None)
	except Exception:
		existing_meta = None
	meta_dict: Dict[str, Any] = {}
	if existing_meta is not None:
		try:
			if isinstance(existing_meta, dict):
				meta_dict.update(existing_meta)
			elif hasattr(existing_meta, "to_plotly_json"):
				m = existing_meta.to_plotly_json()
				if isinstance(m, dict):
					meta_dict.update(m)
			else:
				m = dict(existing_meta)
				if isinstance(m, dict):
					meta_dict.update(m)
		except Exception:
			pass
	meta_dict["tc_freq_switch"] = {
		"label": "Period:",
		"default_index": 0,
		"block_size": n,
		"views": views_meta,
	}
	try:
		per_period_meta = [_layout_meta_to_dict(f) for f in figs]
		key_union: List[str] = sorted({k for md in per_period_meta for k in (md or {}).keys()})
		meta_dict["tc_freq_period_meta"] = {
			"keys": key_union,
			"meta": per_period_meta,
		}
	except Exception:
		pass
	base.update_layout(meta=meta_dict)
	return base


def _build_frequency_switched_html(
	raw_builder: Callable[[List[Any], Optional[str]], str],
	results: List[Any],
	board: Optional[str],
) -> str:
	if not _has_frequency_bundle(results):
		return raw_builder(results, board)

	html_docs: List[str] = []
	for fi in range(3):
		out = raw_builder(_results_view(results, fi), board)
		if not isinstance(out, str):
			raise TypeError(f"Expected str from HTML builder, got {type(out)}")
		html_docs.append(str(out))

	def _extract_title(doc: str) -> Optional[str]:
		try:
			m = re.search(r"<div\s+class=['\"]title['\"]>(.*?)</div>", doc, flags=re.IGNORECASE | re.DOTALL)
			if m:
				return html.unescape(re.sub(r"<[^>]+>", "", m.group(1)).strip())
			m = re.search(r"<title>(.*?)</title>", doc, flags=re.IGNORECASE | re.DOTALL)
			if m:
				return html.unescape(re.sub(r"<[^>]+>", "", m.group(1)).strip())
		except Exception:
			return None
		return None

	page_title = _extract_title(html_docs[0]) or _safe_title(board, "Visual")
	HIDE_HEADER_STYLE = "<style>.header{display:none!important}</style>"
	embedded_docs: List[str] = []
	for d in html_docs:
		low = d.lower()
		needs_hide = ("tabulator" in low)
		if not needs_hide:
			embedded_docs.append(d)
			continue
		if "</head>" in low:
			parts = re.split(r"</head>", d, flags=re.IGNORECASE, maxsplit=1)
			embedded_docs.append(parts[0] + HIDE_HEADER_STYLE + "</head>" + (parts[1] if len(parts) > 1 else ""))
		else:
			embedded_docs.append(HIDE_HEADER_STYLE + d)

	b64_docs = [base64.b64encode(h.encode("utf-8")).decode("ascii") for h in embedded_docs]
	return f"""<!doctype html>
	<html>
	<head>
		<meta charset='utf-8'/>
		<meta name='viewport' content='width=device-width, initial-scale=1'/>
		<title>{html.escape(page_title, quote=True)}</title>
		<style>
			:root {{
				--bg: #ffffff;
				--fg: #111827;
				--muted: #6b7280;
				--border: #e5e7eb;
				--panel: #ffffff;
				--panel2: #f9fafb;
			}}
			body.dark {{
				--bg: #0b1220;
				--fg: #e5e7eb;
				--muted: rgba(229,231,235,0.72);
				--border: rgba(255,255,255,0.18);
				--panel: rgba(17,24,39,0.92);
				--panel2: rgba(17,24,39,0.72);
			}}
			body {{ margin:0; font-family: Segoe UI, Arial, sans-serif; background: var(--bg); color: var(--fg); }}
			.header {{ padding: 10px 12px; border-bottom: 1px solid var(--border); background: var(--panel); display:grid; gap:6px; }}
			.title {{ font-size: 16px; font-weight: 650; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
			.sub {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; color: var(--muted); font-size: 13px; }}
			.tc-mini {{ color: var(--muted); font-size: 12px; }}
			select.tc-menu-btn {{ padding:6px 10px; border:1px solid var(--border); background: var(--panel2); border-radius:10px; cursor:pointer; }}
			.wrap {{ height: calc(100vh - 52px); }}
			iframe {{ width:100%; height:100%; border:0; display:none; }}
			iframe.active {{ display:block; }}
		</style>
	</head>
	<body>
		<div class='header'>
			<div class='sub'>
				<span class='tc-mini'>Period:</span>
				<select class='tc-menu-btn' id='tc-period'>
					<option value='0'>Daily</option>
					<option value='1'>Weekly</option>
					<option value='2'>Monthly</option>
				</select>
				<span>Tip: drag to select, Ctrl+C to copy. Use header filters to search</span>
			</div>
		</div>
		<div class='wrap'>
			<iframe class='active' id='v0' src='data:text/html;base64,{b64_docs[0]}'></iframe>
			<iframe id='v1' src='data:text/html;base64,{b64_docs[1]}'></iframe>
			<iframe id='v2' src='data:text/html;base64,{b64_docs[2]}'></iframe>
		</div>
		<script>
			(function(){{
				const sel = document.getElementById('tc-period');
				const frames = [document.getElementById('v0'), document.getElementById('v1'), document.getElementById('v2')];
				function applyTheme(dark){{
					document.body.classList.toggle('dark', !!dark);
					frames.forEach((f) => {{
						try {{ f && f.contentWindow && f.contentWindow.postMessage({{tc_theme: !!dark}}, '*'); }} catch (e) {{}}
					}});
				}}
				function select(i){{
					sel.value = String(i);
					frames.forEach((f, idx) => f.classList.toggle('active', idx === i));
					try {{
						const savedTheme = localStorage.getItem('tc_theme');
						applyTheme(savedTheme === 'dark');
					}} catch (e) {{
						applyTheme(false);
					}}
					try {{ localStorage.setItem('tc_table_period', String(i)); }} catch (e) {{}}
				}}
				window.addEventListener('message', (ev) => {{
					try {{
								if (!ev || !ev.data) return;
								if (typeof ev.data.tc_theme === 'boolean') applyTheme(ev.data.tc_theme);
					}} catch (e) {{}}
				}});
				sel.addEventListener('change', () => select(parseInt(sel.value || '0', 10) || 0));
				try {{
					const saved = parseInt(localStorage.getItem('tc_table_period') || '0', 10);
					select(isFinite(saved) ? Math.max(0, Math.min(2, saved)) : 0);
				}} catch (e) {{
					select(0);
				}}
			}})();
		</script>
	</body>
	</html>"""
def _wrap_visual_builder(builder: Callable[[List[Any], Optional[str]], Any], *, kind: str) -> Callable[[List[Any], Optional[str]], Any]:
	if kind == "html":
		return lambda results, board=None: _build_frequency_switched_html(builder, results, board)
	return lambda results, board=None: _build_frequency_switched_figure(builder, results, board)


def _as_1d_float(arr: Any, length: Optional[int] = None) -> Optional[np.ndarray]:
	if arr is None:
		return None
	out = np.array(arr, dtype=float)
	if out.ndim != 1:
		return None
	if length is not None and len(out) != length:
		return None
	return out


def _as_2d_float(arr: Any, shape0: Optional[int] = None, shape1: Optional[int] = None) -> Optional[np.ndarray]:
	if arr is None:
		return None
	out = np.array(arr, dtype=float)
	if out.ndim != 2:
		return None
	if shape0 is not None and out.shape[0] != shape0:
		return None
	if shape1 is not None and out.shape[1] != shape1:
		return None
	return out


def _as_str_list(v: Any, length: Optional[int] = None) -> Optional[List[str]]:
	if v is None:
		return None
	if isinstance(v, (list, tuple, np.ndarray)):
		out = [str(x) for x in list(v)]
		if length is not None and len(out) != length:
			return None
		return out
	return None


def _to_datetime_index(lookup_days: Sequence[Any]) -> pd.DatetimeIndex:

	return pd.to_datetime(list(lookup_days), errors="coerce")


def _top_n_indices(values: np.ndarray, n: int) -> np.ndarray:
	if values.size == 0:
		return np.array([], dtype=int)
	n = int(max(1, min(n, values.size)))

	return np.argsort(np.abs(values))[-n:][::-1]


def _shorten_label(text: Any, *, max_len: int = 22) -> str:
	if text is None:
		return ""
	s = str(text)
	s = " ".join(s.split())
	if len(s) <= max_len:
		return s
	if max_len <= 1:
		return "…"
	return s[: max_len - 1] + "…"


def _shrink_legend_names(fig: go.Figure, *, max_len: int = 22) -> None:
	try:
		for tr in list(fig.data) if getattr(fig, "data", None) else []:
			name = getattr(tr, "name", None)
			if not name:
				continue
			short = _shorten_label(name, max_len=max_len)
			if short and short != name:
				tr.name = short
	except Exception:
		return


def _safe_title(board: Optional[str], title: str) -> str:
	if board is None:
		return title
	b = str(board).strip()
	return f"{title} — {b}" if b else title


@dataclass(frozen=True)
class VisualSpec:
	key: str
	label: str
	builder: Callable[[List[Any], Optional[str]], Union[go.Figure, str]]
	kind: str = "figure"


_TEMP_FILES: List[str] = []


def _cleanup_temp_files() -> None:
	import os

	for p in list(_TEMP_FILES):
		try:
			os.remove(p)
		except Exception:
			pass


atexit.register(_cleanup_temp_files)


def _apply_display_layout_defaults(fig: go.Figure) -> None:
	def _has_effective_legend() -> bool:
		try:
			if getattr(fig.layout, "showlegend", None) is False:
				return False
		except Exception:
			pass
		try:
			for tr in list(getattr(fig, "data", []) or []):
				try:
					if getattr(tr, "showlegend", None) is False:
						continue
					name = getattr(tr, "name", None)
					if name is None:
						continue
					if str(name).strip() == "":
						continue
					return True
				except Exception:
					continue
		except Exception:
			return False
		return False

	try:
		layout_json = fig.layout.to_plotly_json()
		margin = dict(layout_json.get("margin") or {})
	except Exception:
		margin = {}

	has_legend = _has_effective_legend()


	if has_legend:
		_shrink_legend_names(fig, max_len=22)


	r_pad = 200 if has_legend else 48
	fig.update_layout(
	 margin=dict(
	  l=max(int(margin.get("l", 0) or 0), 48),
	  r=max(int(margin.get("r", 0) or 0), r_pad),
	  t=max(int(margin.get("t", 0) or 0), 120),
	  b=max(int(margin.get("b", 0) or 0), 56),
	 )
	)


	if has_legend:
		fig.update_layout(
		 legend=dict(
		  orientation="v",
		  x=1.01,
		  xanchor="left",
		  y=1.0,
		  yanchor="top",
		  font=dict(size=11),
		  bgcolor="rgba(0,0,0,0)",
		  borderwidth=0,
		 )
		)
	else:

		try:
			fig.update_layout(showlegend=False)
		except Exception:
			pass



	fig.update_layout(title=dict(x=0.5, xanchor="center", y=0.985, yanchor="top"))



	try:
		ums = list(fig.layout.updatemenus) if getattr(fig.layout, "updatemenus", None) else []
		new_ums = []
		base_y = 1.25
		step = 0.12
		for i, um in enumerate(ums):
			try:
				d = um.to_plotly_json()
			except Exception:
				d = dict(um)
			d["x"] = 0.0
			d["xanchor"] = "left"
			d["y"] = base_y - i * step
			d["yanchor"] = "top"
			d.setdefault("bgcolor", "rgba(241,241,241,0.95)")
			d.setdefault("bordercolor", "rgba(0,0,0,0.18)")
			d.setdefault("borderwidth", 1)
			d.setdefault("font", {"color": "#111827", "size": 12})
			d.setdefault("showactive", True)
			new_ums.append(d)
		fig.update_layout(updatemenus=new_ums)

		# Ensure top margin is large enough for stacked menus.
		try:
			um_count = len(new_ums)
			if um_count > 1:
				m = fig.layout.margin.to_plotly_json() if getattr(fig.layout, "margin", None) else {}
				current_t = int(m.get("t", 0) or 0)
				need_t = 120 + 32 * (um_count - 1)
				if need_t > current_t:
					m["t"] = need_t
					fig.update_layout(margin=m)
		except Exception:
			pass
	except Exception:

		return


def show_figure_in_browser(fig: go.Figure, *, title: str, prefer_data_uri: bool = True) -> None:

	html = _figure_to_html(fig)


	try:
		payload = html.encode("utf-8")

		class _Handler(BaseHTTPRequestHandler):
			def do_GET(self):
				if self.path not in ("/", "/index.html"):
					self.send_response(404)
					self.end_headers()
					return
				self.send_response(200)
				self.send_header("Content-Type", "text/html; charset=utf-8")
				self.send_header("Content-Length", str(len(payload)))
				self.end_headers()
				self.wfile.write(payload)


				threading.Thread(target=self.server.shutdown, daemon=True).start()

			def log_message(self, format, *args):

				return

		httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
		port = httpd.server_address[1]

		threading.Thread(target=httpd.serve_forever, daemon=True).start()
		url = f"http://127.0.0.1:{port}/"


		webbrowser.open(url, new=2, autoraise=True)
		return

	except Exception:

		f = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8")
		try:
			f.write(html)
			f.flush()
			tmp_path = f.name
		finally:
			try:
				f.close()
			except Exception:
				pass

		_TEMP_FILES.append(tmp_path)
		webbrowser.open("file://" + tmp_path, new=2, autoraise=True)

		def _del_later(path: str) -> None:
			import os

			time.sleep(10)
			try:
				os.remove(path)
			except Exception:
				pass

		threading.Thread(target=_del_later, args=(tmp_path,), daemon=True).start()


def _figure_to_html(fig: go.Figure) -> str:
	_apply_display_layout_defaults(fig)




	fragment = pio.to_html(
	 fig,
	 full_html=False,
	 include_plotlyjs="cdn",
	 config={
	  "displayModeBar": True,
	  "scrollZoom": True,
	  "responsive": True,
	 },
	 auto_play=False,
	)
	try:
		page_title = str(getattr(getattr(fig.layout, "title", None), "text", "") or "Plot")
	except Exception:
		page_title = "Plot"
	return _wrap_plotly_fragment_html(fragment=fragment, title=page_title)


def _wrap_plotly_fragment_html(*, fragment: str, title: str) -> str:
	safe_title = html.escape(title or "", quote=True)
	tpl = """<!doctype html>
<html>
<head>
	<meta charset='utf-8'/>
	<meta name='viewport' content='width=device-width, initial-scale=1'/>
	<title>__TITLE__</title>
	<style>
		html, body { height: 100%; }
		body { margin: 0; padding: 10px 12px; font-family: Segoe UI, Arial, sans-serif; background:#fff; color:#111; overflow: hidden; display: flex; flex-direction: row; gap: 12px; height: 100vh; align-items: stretch; }
		body.dark { background:#0f0f10; color:#f0f0f0; }
		#tc-plot { flex: 1 1 auto; min-width: 0; min-height: 0; position: relative; }
		#tc-ui { flex: 0 0 320px; min-width: 260px; max-width: 380px; min-height: 0; overflow: auto; display: none; }
		#tc-ui.has-panels { display: block; }
		#tc-controls {
			position: absolute;
			display: none;
			gap: 8px;
			align-items: center;
			flex-wrap: wrap;
			z-index: 60;
			padding: 6px 8px;
			border-radius: 12px;
			border: 1px solid rgba(0,0,0,0.12);
			background: rgba(247,247,247,0.62);
			-webkit-backdrop-filter: blur(14px) saturate(150%);
			backdrop-filter: blur(14px) saturate(150%);
			box-shadow: 0 10px 30px rgba(0,0,0,0.10);
			max-width: calc(100% - 16px);
			width: min-content;
			top: 10px;

		}
		#tc-controls.has-controls { display: flex; }
		body.dark #tc-controls {
			border-color: rgba(255,255,255,0.16);
			background: rgba(23,23,24,0.55);
			box-shadow: 0 10px 30px rgba(0,0,0,0.45);
		}
		.tv-cont { width: max-content; display: flex; gap: inherit; }
		.tc-menu { position: relative; display: inline-flex; align-items: center; }
		.tc-menu-btn {
			appearance: none;
			border-radius: 999px;
			border: 1px solid rgba(0,0,0,0.18);
			background: rgba(241,241,241,0.70);
			color: #111827;
			padding: 6px 10px;
			font-size: 12px;
			cursor: pointer;
			user-select: none;
			white-space: nowrap;
		}
		body.dark .tc-menu-btn {
			border-color: rgba(255,255,255,0.22);
			background: rgba(30,30,31,0.60);
			color: #f0f0f0;
		}
		.tc-menu-btn.on {
			border-color: rgba(17,99,255,0.55);
			background: rgba(17,99,255,0.16);
		}
		body.dark .tc-menu-btn.on {
			border-color: rgba(120,170,255,0.65);
			background: rgba(120,170,255,0.18);
		}
		.tc-popup {
			position: absolute;
			left: 0;
			top: calc(100% + 6px);
			z-index: 80;
			min-width: 0;
			width: max-content;
			max-width: min(520px, calc(100vw - 24px));
			max-height: 360px;
			overflow: auto;
			padding: 10px 12px;
			border-radius: 12px;
			border: 1px solid rgba(0,0,0,0.12);
			background: rgba(247,247,247,0.92);
			box-shadow: 0 18px 45px rgba(0,0,0,0.18);
			display: none;
		}
		body.dark .tc-popup {
			border-color: rgba(255,255,255,0.16);
			background: rgba(23,23,24,0.92);
			box-shadow: 0 18px 45px rgba(0,0,0,0.45);
		}
		.tc-popup.open { display: block; }
		.tc-row { display:flex; flex-direction: row; gap: 8px; align-items:center; flex-wrap: wrap; }
		.tc-row .tc-btn, .tc-row .tc-pill { width: auto; }
		.tc-mini { font-size: 12px; }
		select.tc-menu-btn {
			padding-right: 28px;
			appearance: none;
			-webkit-appearance: none;
			background-image: linear-gradient(45deg, transparent 50%, currentColor 50%), linear-gradient(135deg, currentColor 50%, transparent 50%);
			background-position: calc(100% - 14px) calc(50% - 2px), calc(100% - 9px) calc(50% - 2px);
			background-size: 5px 5px, 5px 5px;
			background-repeat: no-repeat;
		}
		.tc-inlinebar {
			position: absolute;
			display: flex;
			gap: 8px;
			align-items: center;
			flex-wrap: wrap;
			z-index: 50;
			padding: 6px 8px;
			border-radius: 12px;
			border: 1px solid rgba(0,0,0,0.12);
			background: rgba(247,247,247,0.62);
			-webkit-backdrop-filter: blur(14px) saturate(150%);
			backdrop-filter: blur(14px) saturate(150%);
			box-shadow: 0 10px 30px rgba(0,0,0,0.10);
		}
		body.dark .tc-inlinebar {
			border-color: rgba(255,255,255,0.16);
			background: rgba(23,23,24,0.55);
			box-shadow: 0 10px 30px rgba(0,0,0,0.45);
		}
		.tc-pill {
			appearance: none;
			border-radius: 999px;
			border: 1px solid rgba(0,0,0,0.18);
			background: rgba(241,241,241,0.70);
			color: #111827;
			padding: 6px 10px;
			font-size: 12px;
			cursor: pointer;
			user-select: none;
		}
		body.dark .tc-pill {
			border-color: rgba(255,255,255,0.22);
			background: rgba(30,30,31,0.60);
			color: #f0f0f0;
		}
		.tc-pill.on {
			border-color: rgba(17,99,255,0.55);
			background: rgba(17,99,255,0.16);
		}
		body.dark .tc-pill.on {
			border-color: rgba(120,170,255,0.65);
			background: rgba(120,170,255,0.18);
		}
		.tc-panel {
			display: flex; flex-direction: column; gap: 10px; align-items: stretch;
			padding: 10px 12px;
			border: 1px solid rgba(0,0,0,0.12);
			border-radius: 12px;
			background: rgba(247,247,247,0.62);
			-webkit-backdrop-filter: blur(14px) saturate(150%);
			backdrop-filter: blur(14px) saturate(150%);
			box-shadow: 0 10px 30px rgba(0,0,0,0.10);
			max-width: 100%;
		}
		body.dark .tc-panel { border-color: rgba(255,255,255,0.16); background: rgba(23,23,24,0.55); box-shadow: 0 10px 30px rgba(0,0,0,0.45); }
		.tc-group { display:flex; flex-direction: column; gap: 6px; min-width: 0; }
		.tc-label { font-size: 12px; opacity: 0.85; }
		.tc-select, .tc-input, .tc-btn {
			font-family: Segoe UI, Arial, sans-serif;
			border-radius: 10px;
			border: 1px solid rgba(0,0,0,0.18);
			background: rgba(241,241,241,0.70);
			color: #111827;
			padding: 6px 8px;
			outline: none;
			width: 100%;
		}
		body.dark .tc-select, body.dark .tc-input, body.dark .tc-btn {
			border-color: rgba(255,255,255,0.22);
			background: rgba(30,30,31,0.60);
			color: #f0f0f0;
		}
		.tc-select { height: 132px; }
		.tc-checklist {
			width: 100%;
			max-width: 100%;
			max-height: 180px;
			overflow: auto;
			padding: 8px 10px;
			border-radius: 10px;
			border: 1px solid rgba(0,0,0,0.18);
			background: rgba(255,255,255,0.55);
			display: grid;
			grid-template-columns: 1fr;
			gap: 6px;
		}
		.tc-popup .tc-checklist {
			width: max-content;
			min-width: 220px;
			max-width: min(480px, calc(100vw - 36px));
			overflow-x: auto;
			grid-template-columns: max-content;
		}
		.tc-popup .tc-checkitem { white-space: nowrap; }
		body.dark .tc-checklist {
			border-color: rgba(255,255,255,0.22);
			background: rgba(0,0,0,0.20);
		}
		.tc-checkitem { display:flex; gap: 8px; align-items:center; font-size: 12px; }
		.tc-checkitem input { width: 14px; height: 14px; }
		.tc-cont { width: max-content; display: flex; gap: inherit; align-items: center; }
		.tc-btn { cursor: pointer; }
		.tc-btn:hover { filter: brightness(0.98); }
		body.dark .tc-btn:hover { filter: brightness(1.06); }
		.plotly-graph-div { width: 100% !important; height: 100% !important; }
	</style>
</head>
<body>
	<div id='tc-plot'>
__FRAGMENT__
	</div>
	<div id='tc-ui'></div>

	<script>
		(function() {
			function _downloadBlob(filename, mime, text) {
				try {
					const blob = new Blob([text], { type: mime || 'application/octet-stream' });
					const url = URL.createObjectURL(blob);
					const a = document.createElement('a');
					a.href = url;
					a.download = filename || 'export';
					a.style.display = 'none';
					document.body.appendChild(a);
					a.click();
					setTimeout(() => {
						try { URL.revokeObjectURL(url); } catch (e) {}
						try { a.remove(); } catch (e) {}
					}, 120);
					return true;
				} catch (e) {
					return false;
				}
			}

			function _firstPlotlyDiv() {
				try {
					const divs = Array.from(document.querySelectorAll('.plotly-graph-div'));
					return divs.length ? divs[0] : null;
				} catch (e) {
					return null;
				}
			}

			function _exportCapsPlotly() {
				const div = _firstPlotlyDiv();
				const hasPlot = !!(div && window.Plotly);
				return {
					kind: 'plotly',
					supported: hasPlot,
					image: hasPlot ? ['png', 'jpeg', 'webp', 'svg'] : [],
					data: hasPlot ? ['csv', 'json'] : [],
					other: hasPlot ? ['html'] : ['html'],
					clipboard: hasPlot ? ['csv', 'json'] : [],
				};
			}

			function _plotlyLongCsv(div) {
				function esc(v) {
					const s = (v == null) ? '' : String(v);
					if (/[\\n\\r,\\"]/g.test(s)) return '"' + s.replace(/\\"/g, '""') + '"';
					return s;
				}
				const out = [];
				out.push(['trace_index','trace_name','trace_type','point_index','x','y','z','value'].join(','));
				const data = (div && Array.isArray(div.data)) ? div.data : [];
				for (let ti = 0; ti < data.length; ti++) {
					const t = data[ti] || {};
					const nm = (t.name == null) ? '' : String(t.name);
					const tp = (t.type == null) ? '' : String(t.type);
					if (String(tp).toLowerCase() === 'surface' && Array.isArray(t.z)) {
						const xs = Array.isArray(t.x) ? t.x : [];
						const ys = Array.isArray(t.y) ? t.y : [];
						const z2 = t.z;
						for (let yi = 0; yi < z2.length; yi++) {
							const row = Array.isArray(z2[yi]) ? z2[yi] : [];
							for (let xi = 0; xi < row.length; xi++) {
								const x = (xs.length > xi) ? xs[xi] : xi;
								const y = (ys.length > yi) ? ys[yi] : yi;
								const z = row[xi];
								out.push([ti, esc(nm), esc(tp), (yi * Math.max(1,row.length) + xi), esc(x), esc(y), esc(z), ''].join(','));
							}
						}
						continue;
					}
					const xs = Array.isArray(t.x) ? t.x : [];
					const ys = Array.isArray(t.y) ? t.y : [];
					const vs = Array.isArray(t.values) ? t.values : [];
					const n = Math.max(xs.length, ys.length, vs.length);
					for (let i = 0; i < n; i++) {
						const x = (xs.length > i) ? xs[i] : '';
						const y = (ys.length > i) ? ys[i] : '';
						const v = (vs.length > i) ? vs[i] : '';
						out.push([ti, esc(nm), esc(tp), i, esc(x), esc(y), '', esc(v)].join(','));
					}
				}
				return out.join('\\n');
			}

			async function _copyToClipboard(text) {
				try {
					if (navigator && navigator.clipboard && navigator.clipboard.writeText) {
						await navigator.clipboard.writeText(text);
						return true;
					}
				} catch (e) {}
				return false;
			}

			async function _handleExportMessage(msg) {
				try {
					if (!msg || !msg.action) return;
					if (msg.action === 'caps') {
						try {
							window.parent && window.parent.postMessage({ tc_export_caps: _exportCapsPlotly() }, '*');
						} catch (e) {}
						return;
					}
					if (msg.action !== 'run') return;
					const div = _firstPlotlyDiv();
					const kind = String(msg.kind || '');
					if (kind && kind !== 'plotly') return;
					const scope = String(msg.scope || '');
					const fmt = String(msg.format || '');
					const baseName = (msg.filename == null) ? 'export' : String(msg.filename);
					if (scope === 'image') {
						if (!div || !window.Plotly || !window.Plotly.downloadImage) return;
						try {
							window.Plotly.downloadImage(div, { format: fmt || 'png', filename: baseName, scale: 2 });
						} catch (e) {}
						return;
					}
					if (scope === 'data') {
						if (!div) return;
						if (fmt === 'json') {
							const payload = JSON.stringify({ data: div.data || [], layout: div.layout || {} }, null, 2);
							_downloadBlob(baseName + '.json', 'application/json;charset=utf-8', payload);
							return;
						}
						if (fmt === 'csv') {
							const csv = _plotlyLongCsv(div);
							_downloadBlob(baseName + '.csv', 'text/csv;charset=utf-8', csv);
							return;
						}
					}
					if (scope === 'other') {
						if (fmt === 'html') {
							const html = document.documentElement.outerHTML;
							_downloadBlob(baseName + '.html', 'text/html;charset=utf-8', html);
							return;
						}
					}
					if (scope === 'clipboard') {
						if (!div) return;
						if (fmt === 'json') {
							const payload = JSON.stringify({ data: div.data || [], layout: div.layout || {} }, null, 2);
							await _copyToClipboard(payload);
							return;
						}
						if (fmt === 'csv') {
							const csv = _plotlyLongCsv(div);
							await _copyToClipboard(csv);
							return;
						}
					}
				} catch (e) {}
			}
			function _hasLegend(div) {
				try {
					const layout = (div && div.layout) ? div.layout : {};
					if (layout.showlegend === false) return false;
					const data = (div && Array.isArray(div.data)) ? div.data : [];
					for (let i = 0; i < data.length; i++) {
						const t = data[i] || {};
						if (t.showlegend === false) continue;
						const nm = (t.name == null) ? '' : String(t.name).trim();
						if (!nm) continue;
						return true;
					}
					return false;
				} catch (e) {
					return false;
				}
			}

			function _shortenLabel(s, maxLen) {
				try {
					let out = (s == null) ? '' : String(s);
					out = out.replace(/\\s+/g, ' ').trim();
					if (out.length <= maxLen) return out;
					if (maxLen <= 1) return out.slice(0, maxLen);
					return out.slice(0, maxLen - 1) + '…';
				} catch (e) {
					return '';
				}
			}

			function _hasColorbar(div) {
				try {
					const data = (div && Array.isArray(div.data)) ? div.data : [];
					for (let i = 0; i < data.length; i++) {
						const t = data[i] || {};
						if (t.colorbar) return true;
						if (t.showscale && (t.type || '').toLowerCase() !== 'scatter') return true;
						if (t.marker && t.marker.colorbar) return true;
						if (t.line && t.line.colorbar) return true;
					}
					return false;
				} catch (e) {
					return false;
				}
			}

			function _applyTraceTheme(dark, div) {
				try {
					if (!window.Plotly || !div || !Array.isArray(div.data)) return;
					const idxs = [];
					for (let i = 0; i < div.data.length; i++) {
						const t = div.data[i] || {};
						const nm = (t.name == null) ? '' : String(t.name);
						if (nm === 'Total' || nm === 'Total (100%)') idxs.push(i);
					}
					if (!idxs.length) return;
					const col = dark ? '#e5e7eb' : '#111827';
					window.Plotly.restyle(div, { 'line.color': col }, idxs);
				} catch (e) {}
			}

			function _deepCopy2d(z) {
				if (!Array.isArray(z)) return [];
				return z.map(r => Array.isArray(r) ? r.slice() : []);
			}

			function _ensureUiPanel() {
				return document.getElementById('tc-ui');
			}

			function _ensureControlsBar() {
				const plotWrap = document.getElementById('tc-plot');
				if (!plotWrap) return null;
				let bar = document.getElementById('tc-controls');
				if (bar) return bar;
				bar = document.createElement('div');
				bar.id = 'tc-controls';
				plotWrap.appendChild(bar);
				return bar;
			}

			function _is3dDiv(div) {
				try {
					if (!div) return false;
					const layout = div.layout || {};
					if (layout && (layout.scene || Object.keys(layout).some(k => String(k).startsWith('scene')))) return true;
					const data = Array.isArray(div.data) ? div.data : [];
					for (let i = 0; i < data.length; i++) {
						const t = data[i] || {};
						if (String(t.type || '').toLowerCase() === 'surface') return true;
					}
					return false;
				} catch (e) {
					return false;
				}
			}

			function _positionControls(div) {
				try {
					const plotWrap = document.getElementById('tc-plot');
					const bar = document.getElementById('tc-controls');
					if (!plotWrap || !bar) return;
					const left = 10;
					const top = _is3dDiv(div) ? 10 : 10;
					bar.style.left = left + 'px';
					bar.style.top = top + 'px';
					bar.style.width = 'min-content';
				} catch (e) {}
			}

			function _freqState(div) {
				try {
					const meta = (div && div.layout && div.layout.meta) ? div.layout.meta : {};
					const cfg = meta && meta.tc_freq_switch;
					if (!cfg) return null;
					const views = Array.isArray(cfg.views) ? cfg.views : [];
					if (!views.length) return null;
					div.__tc_freq = div.__tc_freq || {};
					const st = div.__tc_freq;
					st.views = views;
					let bs = Number(cfg.block_size);
					if (!isFinite(bs) || bs <= 0) {
						const v0 = views[0] || {};
						const tot = Array.isArray(v0.visible) ? v0.visible.length : ((div && Array.isArray(div.data)) ? div.data.length : 0);
						bs = Math.max(0, Math.floor(Number(tot || 0) / 3));
					}
					st.blockSize = bs;
					if (!isFinite(Number(st.idx))) {
						const defIdx = (cfg.default_index == null) ? 0 : Number(cfg.default_index);
						st.idx = isFinite(defIdx) ? defIdx : 0;
					}
					st.idx = Math.max(0, Math.min(views.length - 1, Number(st.idx) || 0));
					const pm = meta && meta.tc_freq_period_meta;
					if (pm && typeof pm === 'object') st.periodMeta = pm;
					return st;
				} catch (e) {
					return null;
				}
			}

			function _applyPeriodMeta(div) {
				try {
					const st = _freqState(div);
					if (!st || !st.periodMeta) return;
					const pm = st.periodMeta;
					const keys = Array.isArray(pm.keys) ? pm.keys : [];
					const metaList = Array.isArray(pm.meta) ? pm.meta : [];
					if (!metaList.length) return;
					const sel = metaList[Math.max(0, Math.min(metaList.length - 1, Number(st.idx) || 0))] || {};
					div.layout = div.layout || {};
					div.layout.meta = div.layout.meta || {};
					const m = div.layout.meta;
					const useKeys = keys.length ? keys : Object.keys(sel || {});
					useKeys.forEach((k) => {
						try {
							if (sel && Object.prototype.hasOwnProperty.call(sel, k)) m[k] = sel[k];
							else delete m[k];
						} catch (e) {}
					});
				} catch (e) {}
			}

			function _freqOffset(div) {
				const st = _freqState(div);
				if (!st) return 0;
				const bs = Number(st.blockSize || 0);
				const idx = Number(st.idx || 0);
				return (isFinite(bs) && bs > 0) ? (idx * bs) : 0;
			}

			function _reapplyAfterFreq(div) {
				try {
					const a = div && div.__tc_apply ? div.__tc_apply : null;
					if (!a) return;
					if (typeof a.syncStackMeta === 'function') a.syncStackMeta();
					if (typeof a.syncDistMeta === 'function') a.syncDistMeta();
					if (typeof a.syncSurfaceMeta === 'function') a.syncSurfaceMeta();
					if (typeof a.viewSwitch === 'function') a.viewSwitch();
					if (typeof a.topN === 'function') a.topN();
					if (typeof a.stacked === 'function') a.stacked();
					if (typeof a.finalPnlPick === 'function') a.finalPnlPick();
					if (typeof a.distApply === 'function') a.distApply();
					if (typeof a.surfaceApply === 'function') a.surfaceApply();
				} catch (e) {}
			}

			function _getBaseVisible(div) {
				try {
					const st = _freqState(div);
					if (st && Array.isArray(st.baseVisible) && st.baseVisible.length) return st.baseVisible.slice();
					const out = [];
					const data = (div && Array.isArray(div.data)) ? div.data : [];
					for (let i = 0; i < data.length; i++) {
						const t = data[i] || {};
						out.push((t.visible == null) ? true : t.visible);
					}
					return out;
				} catch (e) {
					return [];
				}
			}

			function _applyFreqIndex(div, idx) {
				try {
					const st = _freqState(div);
					if (!st) return false;
					const views = st.views || [];
					if (!views.length) return false;
					const i = Math.max(0, Math.min(views.length - 1, Number(idx) || 0));
					st.idx = i;
					const v = views[i] || views[0];
					if (v && Array.isArray(v.visible)) {
						const vis = v.visible.slice();
						st.baseVisible = vis.slice();
						try { window.Plotly.restyle(div, { 'visible': vis }); } catch (e) {}
					}
					try {
						const rel = (v && v.relayout) ? v.relayout : {};
						if (rel && typeof rel === 'object') window.Plotly.relayout(div, rel);
					} catch (e) {}
					try { window.Plotly.relayout(div, { 'title.text': '' }); } catch (e) {}
					try { _applyPeriodMeta(div); } catch (e) {}
					try { _reapplyAfterFreq(div); } catch (e) {}
					return true;
				} catch (e) {
					return false;
				}
			}

			function _applyVisibleWithBase(div, mutate) {
				const vis = _getBaseVisible(div);
				if (!vis.length) return;
				try { if (typeof mutate === 'function') mutate(vis); } catch (e) {}
				try { window.Plotly.restyle(div, { 'visible': vis }); } catch (e) {}
				try {
					const st = _freqState(div);
					if (st) st.baseVisible = vis.slice();
				} catch (e) {}
			}

			function _makeFreqSwitchControl(div) {
				try {
					const meta = (div && div.layout && div.layout.meta) ? div.layout.meta : {};
					const cfg = meta && meta.tc_freq_switch;
					if (!cfg) return null;
					const views = Array.isArray(cfg.views) ? cfg.views : [];
					if (!views.length) return null;
					const st = _freqState(div);

					const wrap = document.createElement('div');
					wrap.className = 'tc-menu';
					const lab = document.createElement('span');
					lab.className = 'tc-mini';
					lab.textContent = (cfg.label || 'Period:');
					lab.style.marginRight = '6px';
					const sel = document.createElement('select');
					sel.className = 'tc-menu-btn';
					views.forEach((v, i) => {
						const opt = document.createElement('option');
						opt.value = String(i);
						opt.textContent = (v && v.label) ? String(v.label) : ('P' + String(i + 1));
						sel.appendChild(opt);
					});
					const defIdx = st ? Number(st.idx || 0) : ((cfg.default_index == null) ? 0 : Number(cfg.default_index));
					sel.value = String(isFinite(defIdx) ? defIdx : 0);
					sel.addEventListener('change', () => {
						const idx = Number(sel.value || 0) || 0;
						_applyFreqIndex(div, idx);
						setTimeout(() => _positionControls(div), 0);
					});
					wrap.appendChild(lab);
					wrap.appendChild(sel);
					try {
						if (st && !st._appliedOnce) {
							st._appliedOnce = true;
							setTimeout(() => { try { _applyFreqIndex(div, Number(sel.value || 0) || 0); } catch (e) {} }, 0);
						}
					} catch (e) {}
					return wrap;
				} catch (e) {
					return null;
				}
			}

			function _initViewSwitchSelector(div) {
				try {
					const meta = (div && div.layout && div.layout.meta) ? div.layout.meta : {};
					const cfg = meta && meta.tc_view_switch;
					if (!cfg) return false;
					const views = Array.isArray(cfg.views) ? cfg.views : [];
					if (!views.length) return false;

					const freqWrap = _makeFreqSwitchControl(div);

					const wrap = document.createElement('div');
					wrap.className = 'tc-menu';
					const lab = document.createElement('span');
					lab.className = 'tc-mini';
					lab.textContent = (cfg.label || 'View:');
					lab.style.marginRight = '6px';
					const sel = document.createElement('select');
					sel.className = 'tc-menu-btn';
					views.forEach((v, i) => {
						const opt = document.createElement('option');
						opt.value = String(i);
						opt.textContent = (v && v.label) ? String(v.label) : ('View ' + String(i + 1));
						sel.appendChild(opt);
					});
					const defIdx = (cfg.default_index == null) ? 0 : Number(cfg.default_index);
					sel.value = String(isFinite(defIdx) ? defIdx : 0);

					function _applySelectedView() {
						const idx = Number(sel.value || 0) || 0;
						const v = views[Math.max(0, Math.min(views.length - 1, idx))] || views[0];
						if (!v) return;
						try {
							const st = _freqState(div);
							const bs = st ? Number(st.blockSize || 0) : 0;
							const off = _freqOffset(div);
							if (Array.isArray(v.visible)) {
								if (bs > 0 && v.visible.length === bs) {
									_applyVisibleWithBase(div, (vis) => {
										for (let j = 0; j < bs; j++) {
											const k = off + j;
											if (k >= 0 && k < vis.length) vis[k] = v.visible[j];
										}
									});
								} else {
									if (v.visible.length === ((div.data || []).length)) {
										_applyVisibleWithBase(div, (vis) => {
											for (let j = 0; j < vis.length; j++) vis[j] = v.visible[j];
										});
									}
								}
							}
						} catch (e) {}
						try {
							const rel = (v && v.relayout) ? v.relayout : {};
							if (rel && typeof rel === 'object') window.Plotly.relayout(div, rel);
						} catch (e) {}
						try { window.Plotly.relayout(div, { 'title.text': '' }); } catch (e) {}
						setTimeout(() => _positionControls(div), 0);
					}

					sel.addEventListener('change', _applySelectedView);
					div.__tc_apply = div.__tc_apply || {};
					div.__tc_apply.viewSwitch = _applySelectedView;
					wrap.appendChild(lab);
					wrap.appendChild(sel);
					_addControlItems(div, [freqWrap, wrap].filter(Boolean));
					return true;
				} catch (e) {
					return false;
				}
			}

			function _initTopNSelector(div) {
				try {
					const meta = (div && div.layout && div.layout.meta) ? div.layout.meta : {};
					const cfg = meta && meta.tc_topn;
					if (!cfg) return false;
					const freqWrap = _makeFreqSwitchControl(div);
					const traceStart = Number(cfg.trace_start);
					const traceCount = Number(cfg.trace_count);
					const rank = Array.isArray(cfg.rank) ? cfg.rank.map(v => Number(v)) : [];
					if (!isFinite(traceStart) || !isFinite(traceCount) || traceCount <= 0) return false;
					if (!rank.length) return false;

					const options = Array.isArray(cfg.options) ? cfg.options : [12, 25, 0];
					const label = cfg.label || 'Show:';

					function _topIdx(n) {
						const pairs = rank.map((v, i) => ({ i, v: Math.abs(Number(v) || 0) }));
						pairs.sort((a, b) => b.v - a.v);
						const k = (n == null || n <= 0) ? pairs.length : Math.max(1, Math.min(pairs.length, n));
						const out = new Set();
						for (let j = 0; j < k; j++) out.add(pairs[j].i);
						return out;
					}

					function _apply(n) {
						const top = _topIdx(n);
						const st = _freqState(div);
						const bs = st ? Number(st.blockSize || 0) : 0;
						const off = _freqOffset(div);
						_applyVisibleWithBase(div, (vis) => {
							const start = (bs > 0) ? (off + traceStart) : traceStart;
							for (let i = 0; i < traceCount; i++) {
								const k = start + i;
								if (k >= 0 && k < vis.length) {
									vis[k] = top.has(i) ? true : 'legendonly';
								}
							}
						});
					}

					const wrap = document.createElement('div');
					wrap.className = 'tc-menu';
					const lab = document.createElement('span');
					lab.className = 'tc-mini';
					lab.textContent = label;
					lab.style.marginRight = '6px';
					const sel = document.createElement('select');
					sel.className = 'tc-menu-btn';
					options.forEach((n) => {
						const opt = document.createElement('option');
						opt.value = String(n);
						opt.textContent = (n && n > 0) ? ('Top ' + String(n)) : 'All';
						sel.appendChild(opt);
					});
					const defN = Number(cfg.default_n);
					sel.value = String(isFinite(defN) ? defN : (options[0] || 12));
					div.__tc_apply = div.__tc_apply || {};
					div.__tc_apply.topN = () => {
						try {
							const nn = Number(sel.value || cfg.default_n || 10) || 10;
							_apply(nn);
						} catch (e) {}
					};
					sel.addEventListener('change', () => {
						const n = Number(sel.value);
						_apply(isFinite(n) ? n : 12);
					});
					wrap.appendChild(lab);
					wrap.appendChild(sel);
					_addControlItems(div, [freqWrap, wrap].filter(Boolean));
					return true;
				} catch (e) {
					return false;
				}
			}

			function _initFreqSelector(div) {
				try {
					const freqWrap = _makeFreqSwitchControl(div);
					if (!freqWrap) return false;
					_addControlItems(div, [freqWrap]);
					return true;
				} catch (e) {
					return false;
				}
			}

			function _closeAllPopups(exceptEl) {
				try {
					Array.from(document.querySelectorAll('.tc-popup.open')).forEach(p => {
						if (exceptEl && p === exceptEl) return;
						p.classList.remove('open');
					});
				} catch (e) {}
			}

			function _makeMenu(labelText, contentEl) {
				const wrap = document.createElement('div');
				wrap.className = 'tc-menu';
				const btn = document.createElement('button');
				btn.type = 'button';
				btn.className = 'tc-menu-btn';
				btn.textContent = labelText;
				const pop = document.createElement('div');
				pop.className = 'tc-popup';
				pop.appendChild(contentEl);
				btn.addEventListener('click', (ev) => {
					try { ev && ev.stopPropagation && ev.stopPropagation(); } catch (e) {}
					const isOpen = pop.classList.contains('open');
					_closeAllPopups(isOpen ? null : pop);
					pop.classList.toggle('open');
				});
				wrap.appendChild(btn);
				wrap.appendChild(pop);
				document.addEventListener('click', (ev) => {
					const t = ev && ev.target;
					if (!t) return;
					if (wrap.contains(t)) return;
					pop.classList.remove('open');
				}, true);
				return { wrap, btn, pop };
			}

			function _addControlItems(div, items) {
				const bar = _ensureControlsBar();
				if (!bar) return null;
				bar.innerHTML = '';
				(items || []).forEach(it => { if (it) bar.appendChild(it); });
				bar.classList.add('has-controls');
				setTimeout(() => _positionControls(div), 80);
				window.addEventListener('resize', () => _positionControls(div));
				try { if (div && div.on) div.on('plotly_buttonclicked', () => _positionControls(div)); } catch (e) {}
				return bar;
			}

			function _addDistribFieldControls(div, items) {
				const bar = _ensureControlsBar();
				if (!bar) return null;
				bar.innerHTML = '';
				bar.classList.add('has-controls');
				bar.style.left = '10px';
				bar.style.top = '10px';
				bar.style.width = 'min-content';

				const cont = document.createElement('div');
				cont.className = 'tc-cont';
				cont.style.width = 'max-content';
				cont.style.display = 'flex';
				cont.style.gap = 'inherit';
				(items || []).forEach(it => { if (it) cont.appendChild(it); });
				bar.appendChild(cont);

				setTimeout(() => _positionControls(div), 80);
				window.addEventListener('resize', () => _positionControls(div));
				try { if (div && div.on) div.on('plotly_buttonclicked', () => _positionControls(div)); } catch (e) {}
				return bar;
			}

			function _add3dChoiceFieldControls(div, items) {
				const bar = _ensureControlsBar();
				if (!bar) return null;
				bar.innerHTML = '';
				bar.classList.add('has-controls');
				bar.style.left = '10px';
				bar.style.top = '10px';
				bar.style.width = 'min-content';

				const cont = document.createElement('div');
				cont.className = 'tc-cont';
				cont.style.width = 'max-content';
				cont.style.display = 'flex';
				cont.style.gap = 'inherit';
				(items || []).forEach(it => { if (it) cont.appendChild(it); });
				bar.appendChild(cont);

				setTimeout(() => _positionControls(div), 80);
				window.addEventListener('resize', () => _positionControls(div));
				try { if (div && div.on) div.on('plotly_buttonclicked', () => _positionControls(div)); } catch (e) {}
				return bar;
			}

			function _clearUi() {
				try {
					const host = _ensureUiPanel();
					if (host) {
						host.innerHTML = '';
						host.classList.remove('has-panels');
					}
				} catch (e) {}
				try {
					const bar = document.getElementById('tc-controls');
					if (bar) {
						bar.innerHTML = '';
						bar.classList.remove('has-controls');
					}
				} catch (e) {}
				try {
					const ib = document.getElementById('tc-inline-stack');
					if (ib && ib.parentNode) ib.parentNode.removeChild(ib);
				} catch (e) {}
			}

			function _makeGroup(labelText, controlEl) {
				const g = document.createElement('div');
				g.className = 'tc-group';
				const lab = document.createElement('div');
				lab.className = 'tc-label';
				lab.textContent = labelText;
				g.appendChild(lab);
				g.appendChild(controlEl);
				return g;
			}

			function _makeButton(text, onClick) {
				const b = document.createElement('button');
				b.type = 'button';
				b.className = 'tc-btn';
				b.textContent = text;
				b.addEventListener('click', onClick);
				return b;
			}

			function _selectedValues(selectEl) {
				const out = [];
				for (const opt of Array.from(selectEl.options || [])) {
					if (opt.selected) out.push(opt.value);
				}
				return out;
			}

			function _makeChecklist(values, selectedSet, labelMaxLen) {
				const box = document.createElement('div');
				box.className = 'tc-checklist';
				const maxLen = (labelMaxLen == null) ? 40 : Number(labelMaxLen);
				(values || []).forEach(v => {
					const lab = document.createElement('label');
					lab.className = 'tc-checkitem';
					const cb = document.createElement('input');
					cb.type = 'checkbox';
					cb.value = String(v);
					cb.checked = selectedSet ? selectedSet.has(String(v)) : false;
					const sp = document.createElement('span');
					sp.textContent = _shortenLabel(v, maxLen);
					sp.title = String(v);
					lab.appendChild(cb);
					lab.appendChild(sp);
					box.appendChild(lab);
				});
				return box;
			}

			function _checkedValues(checklistEl) {
				try {
					return Array.from(checklistEl.querySelectorAll('input[type="checkbox"]:checked')).map(el => String(el.value));
				} catch (e) {
					return [];
				}
			}

			function _setAllChecks(checklistEl, on) {
				try {
					Array.from(checklistEl.querySelectorAll('input[type="checkbox"]')).forEach(el => { el.checked = !!on; });
				} catch (e) {}
			}

			function _initFinalPnlBarSelector(div) {
				try {
					const title = (div.__tc_title_text != null) ? String(div.__tc_title_text) : ((div.layout && div.layout.title && div.layout.title.text) ? String(div.layout.title.text) : '');
					if (!title.toLowerCase().includes('final pnl')) return false;
					if (!Array.isArray(div.data) || div.data.length !== 1) return false;
					const tr = div.data[0] || {};
					if (String(tr.type || '').toLowerCase() !== 'bar') return false;
					const x = Array.isArray(tr.x) ? tr.x.map(v => String(v)) : [];
					const y = Array.isArray(tr.y) ? tr.y.map(v => Number(v)) : [];
					if (x.length < 2 || x.length !== y.length) return false;
					div.__tc_base = div.__tc_base || {};
					if (!div.__tc_base.bar) div.__tc_base.bar = { x: x.slice(), y: y.slice() };
					const selected = new Set(div.__tc_base.bar.x.map(v => String(v)));
					const checklist = _makeChecklist(div.__tc_base.bar.x, selected, 34);

					const applyBtn = _makeButton('Apply', () => {
						const picked = _checkedValues(checklist);
						const baseX = div.__tc_base.bar.x;
						const baseY = div.__tc_base.bar.y;
						const pickSet = new Set(picked);
						const x2 = [];
						const y2 = [];
						for (let i = 0; i < baseX.length; i++) {
							if (picked.length === 0 || pickSet.has(baseX[i])) {
								x2.push(baseX[i]);
								y2.push(baseY[i]);
							}
						}
						const colors2 = y2.map(v => (v >= 0) ? '#2ca02c' : '#d62728');
						try {
							window.Plotly.restyle(div, { 'x': [x2], 'y': [y2], 'marker.color': [colors2] }, [0]);
							window.Plotly.Plots.resize(div);
						} catch (e) {}
					});

					const allBtn = document.createElement('button');
					allBtn.type = 'button';
					allBtn.className = 'tc-menu-btn';
					allBtn.textContent = 'All';
					allBtn.addEventListener('click', () => { _setAllChecks(checklist, true); });
					const noneBtn = document.createElement('button');
					noneBtn.type = 'button';
					noneBtn.className = 'tc-menu-btn';
					noneBtn.textContent = 'None';
					noneBtn.addEventListener('click', () => { _setAllChecks(checklist, false); });

					const popupBody = document.createElement('div');
					popupBody.style.display = 'grid';
					popupBody.style.gap = '10px';
					popupBody.appendChild(checklist);
					const row = document.createElement('div');
					row.className = 'tc-row';
					row.appendChild(applyBtn);
					row.appendChild(allBtn);
					row.appendChild(noneBtn);
					popupBody.appendChild(row);

					const menu = _makeMenu('Final PnL: pick tickers/types', popupBody);
					const freqWrap = _makeFreqSwitchControl(div);
					_addControlItems(div, [freqWrap, menu.wrap].filter(Boolean));
					return true;
				} catch (e) {
					return false;
				}
			}

			function _initDistributionBarSelector(div) {
				try {
					const meta = (div.layout && div.layout.meta && div.layout.meta.tc_dist_full) ? div.layout.meta.tc_dist_full : null;
					if (!meta) return false;
					if (!Array.isArray(div.data) || div.data.length < 1) return false;
					const off0 = _freqOffset(div);
					const tr = div.data[off0] || div.data[0] || {};
					if (String(tr.type || '').toLowerCase() !== 'bar') return false;
					const labels = Array.isArray(meta.labels) ? meta.labels.map(v => String(v)) : [];
					const dates = Array.isArray(meta.dates) ? meta.dates.map(v => String(v)) : [];
					const mat = Array.isArray(meta.mat) ? meta.mat : [];
					if (!labels.length || !dates.length || !Array.isArray(mat) || !Array.isArray(mat[0] || [])) return false;

					div.__tc_base = div.__tc_base || {};
					if (!div.__tc_base.dist) {
						div.__tc_base.dist = { labels: labels, dates: dates, mat: mat };
					}

					const base = div.__tc_base.dist;
					const dateSel = document.createElement('select');
					dateSel.className = 'tc-menu-btn';
					for (const d of base.dates) {
						const opt = document.createElement('option');
						opt.value = d;
						opt.textContent = d;
						dateSel.appendChild(opt);
					}
					dateSel.value = base.dates[base.dates.length - 1];

					function _syncDistMeta() {
						try {
							const m2 = (div.layout && div.layout.meta && div.layout.meta.tc_dist_full) ? div.layout.meta.tc_dist_full : null;
							if (!m2) return;
							const labels2 = Array.isArray(m2.labels) ? m2.labels.map(v => String(v)) : base.labels;
							const dates2 = Array.isArray(m2.dates) ? m2.dates.map(v => String(v)) : base.dates;
							const mat2 = Array.isArray(m2.mat) ? m2.mat : base.mat;
							base.labels = labels2;
							base.dates = dates2;
							base.mat = mat2;
							const prev = String(dateSel.value || '');
							dateSel.innerHTML = '';
							for (const d of base.dates) {
								const opt = document.createElement('option');
								opt.value = d;
								opt.textContent = d;
								dateSel.appendChild(opt);
							}
							if (prev && base.dates.indexOf(prev) >= 0) dateSel.value = prev;
							else dateSel.value = base.dates[base.dates.length - 1];
						} catch (e) {}
					}

					const selected = new Set();
					const checklist = _makeChecklist(base.labels, selected, 34);

					function _applyDist() {
						const picked = _checkedValues(checklist);
						const pickSet = new Set(picked);
						const date = String(dateSel.value || base.dates[base.dates.length - 1]);
						let di = base.dates.indexOf(date);
						if (di < 0) di = base.dates.length - 1;

						const rows = [];
						for (let i = 0; i < base.labels.length; i++) {
							const lab = base.labels[i];
							if (picked.length > 0 && !pickSet.has(lab)) continue;
							const row = (Array.isArray(base.mat[i])) ? base.mat[i] : [];
							rows.push({ x: lab, y: Number(row[di]) });
						}
						rows.sort((a, b) => (a.y - b.y));
						const x2 = rows.map(r => r.x);
						const y2 = rows.map(r => r.y);
						const colors2 = y2.map(v => (v >= 0) ? '#2ca02c' : '#d62728');
						let meanIdx = -1;
						let medIdx = -1;
						try {
							const st = _freqState(div);
							const bs = st ? Number(st.blockSize || 0) : 0;
							const off = _freqOffset(div);
							const end = (bs > 0) ? Math.min((div.data || []).length, off + bs) : (div.data || []).length;
							for (let i = Math.max(0, off + 1); i < end; i++) {
								const tt = div.data[i] || {};
								if (String(tt.type || '').toLowerCase() !== 'scatter') continue;
								const nm = String(tt.name || '').toLowerCase();
								if (nm.includes('mean')) meanIdx = i;
								if (nm.includes('median')) medIdx = i;
							}
						} catch (e) {}
						let mean = 0;
						let med = 0;
						try {
							if (y2.length) {
								mean = y2.reduce((a, b) => a + Number(b), 0) / y2.length;
								const ys = y2.map(v => Number(v)).slice().sort((a, b) => a - b);
								const n = ys.length;
								med = (n % 2) ? ys[(n - 1) / 2] : 0.5 * (ys[n / 2 - 1] + ys[n / 2]);
							}
						} catch (e) {}
						const meanY = x2.map(_ => mean);
						const medY = x2.map(_ => med);
						try {
							const off = _freqOffset(div);
							window.Plotly.restyle(div, {
								'x': [x2],
								'y': [y2],
								'marker.color': [colors2],
								'hovertemplate': [`%{x}<br>Date=${date}<br>PnL=%{y:.2f}<extra></extra>`],
							}, [off]);
							if (meanIdx >= 0) {
								window.Plotly.restyle(div, {
									'x': [x2],
									'y': [meanY],
									'hovertemplate': [`Mean=%{y:.2f}<extra></extra>`],
								}, [meanIdx]);
							}
							if (medIdx >= 0) {
								window.Plotly.restyle(div, {
									'x': [x2],
									'y': [medY],
									'hovertemplate': [`Median=%{y:.2f}<extra></extra>`],
								}, [medIdx]);
							}
							window.Plotly.Plots.resize(div);
						} catch (e) {}
					}

					const applyBtn = _makeButton('Apply', _applyDist);
					const resetBtn = _makeButton('Reset', () => {
						dateSel.value = base.dates[base.dates.length - 1];
						_setAllChecks(checklist, false);
						_applyDist();
					});

					const allBtn = document.createElement('button');
					allBtn.type = 'button';
					allBtn.className = 'tc-menu-btn';
					allBtn.textContent = 'All';
					allBtn.addEventListener('click', () => { _setAllChecks(checklist, true); });
					const noneBtn = document.createElement('button');
					noneBtn.type = 'button';
					noneBtn.className = 'tc-menu-btn';
					noneBtn.textContent = 'None';
					noneBtn.addEventListener('click', () => { _setAllChecks(checklist, false); });

					const pickPopup = document.createElement('div');
					pickPopup.style.display = 'grid';
					pickPopup.style.gap = '10px';
					pickPopup.appendChild(checklist);
					const row = document.createElement('div');
					row.className = 'tc-row';
					row.appendChild(allBtn);
					row.appendChild(noneBtn);
					pickPopup.appendChild(row);
					const pickMenu = _makeMenu('Pick tickers/types', pickPopup);

					const dateWrap = document.createElement('div');
					dateWrap.className = 'tc-menu';
					const dateLab = document.createElement('span');
					dateLab.className = 'tc-mini';
					dateLab.textContent = 'Date:';
					dateLab.style.marginRight = '6px';
					dateWrap.appendChild(dateLab);
					dateWrap.appendChild(dateSel);

					applyBtn.className = 'tc-menu-btn';
					resetBtn.className = 'tc-menu-btn';
					const freqWrap = _makeFreqSwitchControl(div);
					div.__tc_apply = div.__tc_apply || {};
					div.__tc_apply.syncDistMeta = _syncDistMeta;
					div.__tc_apply.distApply = () => { try { _syncDistMeta(); } catch (e) {} try { _applyDist(); } catch (e) {} };
					_addDistribFieldControls(div, [freqWrap, dateWrap, pickMenu.wrap, applyBtn, resetBtn].filter(Boolean));
					return true;
				} catch (e) {
					return false;
				}
			}

			function _initStackedSelector(div) {
				try {
					let meta = (div.layout && div.layout.meta && div.layout.meta.tc_stack_full) ? div.layout.meta.tc_stack_full : null;
					if (!meta) return false;
					let isPercent = !!meta.percent;
					let tm = meta.trace_map || {};
					let compCount = Number(tm.comp_count || 0);
					if (!compCount || !Array.isArray(meta.x) || meta.x.length < 2) return false;
					if (!Array.isArray(meta.values) || !Array.isArray(meta.values[0] || [])) return false;
					div.__tc_base = div.__tc_base || {};
					if (!div.__tc_base.stack) {
						div.__tc_base.stack = {
							x: meta.x.map(v => String(v)),
							labels: (meta.labels || []).map(v => String(v)),
							values: meta.values,
							cash: meta.cash || [],
							credit_neg: meta.credit_neg || [],
							total: meta.total || [],
							trace_map: tm,
							percent: isPercent,
						};
					}

					const base = div.__tc_base.stack;

					function _syncStackMeta() {
						try {
							const m2 = (div.layout && div.layout.meta && div.layout.meta.tc_stack_full) ? div.layout.meta.tc_stack_full : null;
							if (!m2) return;
							meta = m2;
							isPercent = !!meta.percent;
							tm = meta.trace_map || {};
							compCount = Number(tm.comp_count || 0);
							base.x = (meta.x || []).map(v => String(v));
							base.labels = (meta.labels || []).map(v => String(v));
							base.values = meta.values;
							base.cash = meta.cash || [];
							base.credit_neg = meta.credit_neg || [];
							base.total = meta.total || [];
							base.trace_map = tm;
							base.percent = isPercent;
						} catch (e) {}
					}
					const absSec = document.createElement('input');
					absSec.type = 'checkbox';
					absSec.checked = false;
					const absCred = document.createElement('input');
					absCred.type = 'checkbox';
					absCred.checked = false;

					function _makeToggleBtn(label, chk) {
						const b = document.createElement('button');
						b.type = 'button';
						b.className = 'tc-menu-btn' + (chk.checked ? ' on' : '');
						b.setAttribute('aria-pressed', chk.checked ? 'true' : 'false');
						b.textContent = label;
						b.addEventListener('click', () => {
							chk.checked = !chk.checked;
							b.className = 'tc-menu-btn' + (chk.checked ? ' on' : '');
							b.setAttribute('aria-pressed', chk.checked ? 'true' : 'false');
							_applyStack();
						});
						return b;
					}

					function _makeIndexSelection(n, maxN) {
						const m = Math.max(10, Math.min(n, maxN));
						if (maxN <= m) return Array.from({length: maxN}, (_, i) => i);
						const stride = Math.max(1, Math.ceil(maxN / m));
						const idx = [];
						for (let i = 0; i < maxN; i += stride) idx.push(i);
						if (idx[idx.length - 1] !== (maxN - 1)) idx.push(maxN - 1);
						return idx;
					}

					function _inferStepMs(xArr) {
						try {
							const ts = (xArr || []).map(v => Date.parse(String(v))).filter(v => Number.isFinite(v));
							if (ts.length < 2) return 24 * 60 * 60 * 1000;
							ts.sort((a, b) => a - b);
							const diffs = [];
							for (let i = 1; i < ts.length; i++) {
								const d = ts[i] - ts[i - 1];
								if (d > 0) diffs.push(d);
							}
							if (!diffs.length) return 24 * 60 * 60 * 1000;
							diffs.sort((a, b) => a - b);
							const mid = Math.floor(diffs.length / 2);
							const step = diffs[mid] || diffs[0];
							return Math.max(60 * 60 * 1000, Math.min(step, 90 * 24 * 60 * 60 * 1000));
						} catch (e) {
							return 24 * 60 * 60 * 1000;
						}
					}

					function _inferLocalWidthsMs(xArr, fallbackStepMs) {
						try {
							const ts = (xArr || []).map(v => Date.parse(String(v)));
							if (!ts.length) return null;
							for (let i = 0; i < ts.length; i++) {
								if (!Number.isFinite(ts[i])) return null;
							}
							const n = ts.length;
							const step = Number.isFinite(fallbackStepMs) ? fallbackStepMs : (24 * 60 * 60 * 1000);
							const out = new Array(n);
							for (let i = 0; i < n; i++) {
								let prevGap = null;
								let nextGap = null;
								if (i > 0) {
									const g = ts[i] - ts[i - 1];
									if (g > 0) prevGap = g;
								}
								if (i < n - 1) {
									const g = ts[i + 1] - ts[i];
									if (g > 0) nextGap = g;
								}
								let local = step;
								if (prevGap != null && nextGap != null) local = Math.min(prevGap, nextGap);
								else if (prevGap != null) local = prevGap;
								else if (nextGap != null) local = nextGap;
								local = Math.max(60 * 60 * 1000, Math.min(local, 90 * 24 * 60 * 60 * 1000));
								out[i] = local;
							}
							return out;
						} catch (e) {
							return null;
						}
					}

					function _isBarMode() {
						try {
							const bi = Number(tm.bar_credit);
							if (!Number.isFinite(bi)) return false;
							const off = _freqOffset(div);
							const k = off + bi;
							const t = (div.data && div.data[k]) ? div.data[k] : null;
							return !!(t && (t.visible === true));
						} catch (e) { return false; }
					}

					let viewMode = _isBarMode() ? 'bar' : 'area';
					function _applyView(mode) {
						try {
							try { _syncStackMeta(); } catch (e) {}
							const vcfg = meta.views || {};
							const v = (mode === 'bar') ? (vcfg.bar || null) : (vcfg.area || null);
							if (v && Array.isArray(v.visible)) {
								const st = _freqState(div);
								const bs = st ? Number(st.blockSize || 0) : 0;
								const off = _freqOffset(div);
								if (bs > 0) {
									let mask = v.visible;
									if (mask.length === ((div.data || []).length)) mask = mask.slice(off, off + bs);
									if (mask.length === bs) {
										_applyVisibleWithBase(div, (vis) => {
											for (let j = 0; j < bs; j++) {
												const k = off + j;
												if (k >= 0 && k < vis.length) vis[k] = mask[j];
											}
										});
									} else {
										_applyVisibleWithBase(div, (vis) => {});
									}
								} else {
									window.Plotly.restyle(div, { 'visible': v.visible });
								}
							}
							if (v && v.relayout && typeof v.relayout === 'object') window.Plotly.relayout(div, v.relayout);
						} catch (e) {}
					}

					function _applyStack() {
						try { _syncStackMeta(); } catch (e) {}
						const useAbsSec = !!absSec.checked;
						const useAbsCred = !!absCred.checked;
						const barMode = _isBarMode();
						const slotPx = 28;
						let plotW = 0;
						try { plotW = (div && div.getBoundingClientRect) ? div.getBoundingClientRect().width : 0; } catch (e) { plotW = 0; }
						if (!plotW || plotW < 240) {
							try {
								const plotWrap = document.getElementById('tc-plot');
								plotW = plotWrap ? plotWrap.clientWidth : 0;
							} catch (e) { plotW = 0; }
						}
						let pick = null;
						if (barMode) {
							pick = Array.from({length: base.x.length}, (_, i) => i);
						} else {
							pick = Array.from({length: base.x.length}, (_, i) => i);
						}
						const x2 = pick.map(i => base.x[i]);
						const stepMs = _inferStepMs(x2);
						const localSteps = _inferLocalWidthsMs(x2, stepMs);

						const compY = [];
						for (let ci = 0; ci < compCount; ci++) {
							const row = Array.isArray(base.values[ci]) ? base.values[ci] : [];
							const out = pick.map(i => {
								const v = Number(row[i]);
								if (useAbsSec) return Math.abs(v);
								return v;
							});
							compY.push(out);
						}
						const cashRow = Array.isArray(base.cash) ? base.cash : [];
						const cashY0 = pick.map(i => Number(cashRow[i]));
						const crRow = Array.isArray(base.credit_neg) ? base.credit_neg : [];
						const crY0 = pick.map(i => {
							const v = Number(crRow[i]);
							return useAbsCred ? Math.abs(v) : v;
						});

						let compYUse = compY;
						let cashY = cashY0;
						let crY = crY0;
						let totY = [];
						if (base.percent) {
							// Percent scaling:
							// - If Abs credit is OFF: negatives do not affect scaling unless credit magnitude
							//   exceeds negative securities magnitude; that excess reduces denom => positives can exceed 100.
							// - If Abs credit is ON: credit becomes positive overlay and participates in denom,
							//   recombining everything back to 100.
							const denom = pick.map((_, k) => {
								let pos = Math.max(0, Number(cashY0[k]));
								let negAbs = 0;
								for (let ci = 0; ci < compCount; ci++) {
									const v = Number(compY[ci][k]);
									pos += Math.max(0, v);
									negAbs += Math.max(0, -v);
								}
								const credAbs = Math.abs(Number(crRow[pick[k]]));
								if (useAbsCred) {
									// Abs credit is an overlay. Keep the positive stack normalized to 100%
									// (do not include credit in denom), otherwise the stack drops below 100.
									return pos;
								}
								const excess = Math.max(0, credAbs - negAbs);
								return pos - excess;
							});
							const safeDen = denom.map(v => (Math.abs(v) < 1e-12) ? 1.0 : v);
							compYUse = compY.map(row => row.map((v, k) => 100.0 * Number(v) / safeDen[k]));
							cashY = cashY0.map((v, k) => 100.0 * Number(v) / safeDen[k]);
							crY = crY0.map((v, k) => 100.0 * Number(v) / safeDen[k]);
							// When denom is ~0, show zeros (and keep Total=100%).
							for (let k = 0; k < denom.length; k++) {
								if (Math.abs(denom[k]) < 1e-12) {
									for (let ci = 0; ci < compCount; ci++) compYUse[ci][k] = 0;
									cashY[k] = 0;
									crY[k] = 0;
								}
							}
							totY = pick.map(_ => 100.0);
						} else {
							const totRow = Array.isArray(base.total) ? base.total : [];
							totY = pick.map(i => {
								let t = Number(totRow[i]);
								let adj = 0;
								if (useAbsSec) {
									let negSum = 0;
									for (let ci = 0; ci < compCount; ci++) {
										const row = Array.isArray(base.values[ci]) ? base.values[ci] : [];
										const v = Number(row[i]);
										if (v < 0) negSum += v;
									}
									adj += -2 * negSum;
								}
								if (useAbsCred) {
									// Abs credit should increase Total only by the part of credit that is larger
									// than absolute negative securities (overlay semantics; avoids double counting).
									let negAbs = 0;
									for (let ci = 0; ci < compCount; ci++) {
										const row = Array.isArray(base.values[ci]) ? base.values[ci] : [];
										const v = Number(row[i]);
										negAbs += Math.max(0, -v);
									}
									const credAbs = Math.abs(Number(crRow[i]));
									const excess = Math.max(0, credAbs - negAbs);
									adj += excess;
								}
								return t + adj;
							});
						}

						function _p(v) { const n = Number(v); return (n > 0) ? n : 0; }
						function _n(v) { const n = Number(v); return (n < 0) ? n : 0; }
						const cashNeg = cashY.map(_n);
						const cashPos = cashY.map(_p);

						const idxs = [];
						const xArr = [];
						const yArr = [];
						const off = _freqOffset(div);
						// area: cash_neg, comps_neg..., credit_overlay, cash_pos, comps_pos...
						idxs.push(off + Number(tm.area_cash_neg)); xArr.push(x2); yArr.push(cashNeg);
						for (let ci = 0; ci < compCount; ci++) {
							idxs.push(off + Number(tm.area_comp_neg_start) + ci);
							xArr.push(x2);
							yArr.push(compYUse[ci].map(_n));
						}
						idxs.push(off + Number(tm.area_credit)); xArr.push(x2); yArr.push(crY);
						idxs.push(off + Number(tm.area_cash_pos)); xArr.push(x2); yArr.push(cashPos);
						for (let ci = 0; ci < compCount; ci++) {
							idxs.push(off + Number(tm.area_comp_pos_start) + ci);
							xArr.push(x2);
							yArr.push(compYUse[ci].map(_p));
						}
						idxs.push(off + Number(tm.overlay)); xArr.push(x2); yArr.push(totY);
						// bar: cash, comps..., credit overlay
						idxs.push(off + Number(tm.bar_cash)); xArr.push(x2); yArr.push(cashY);
						for (let ci = 0; ci < compCount; ci++) { idxs.push(off + Number(tm.bar_comp_start) + ci); xArr.push(x2); yArr.push(compYUse[ci]); }
						idxs.push(off + Number(tm.bar_credit)); xArr.push(x2); yArr.push(crY);
						const barIdxs = [];
						barIdxs.push(off + Number(tm.bar_cash));
						for (let ci = 0; ci < compCount; ci++) barIdxs.push(off + Number(tm.bar_comp_start) + ci);
						barIdxs.push(off + Number(tm.bar_credit));

						try {
							window.Plotly.restyle(div, { 'x': xArr, 'y': yArr }, idxs);
							if (barMode) {
								const wScale = 0.78;
								const w = (localSteps && Array.isArray(localSteps) && localSteps.length === x2.length)
									? localSteps.map(ms => wScale * ms)
									: (wScale * stepMs);
								window.Plotly.restyle(div, { 'width': w }, barIdxs);
							} else {
								window.Plotly.restyle(div, { 'width': null }, barIdxs);
							}
							window.Plotly.relayout(div, {
								'bargap': barMode ? 0.22 : 0.4,
								'bargroupgap': 0.0,
								'xaxis.type': 'date',
								'xaxis.rangeslider.visible': !barMode,
							});
							window.Plotly.Plots.resize(div);
						} catch (e) {}
					}

					let _debT = null;
					function _debouncedApply() {
						try { if (_debT) clearTimeout(_debT); } catch (e) {}
						_debT = setTimeout(_applyStack, 80);
					}
					window.addEventListener('resize', _debouncedApply);
					try { if (div && div.on) div.on('plotly_buttonclicked', _debouncedApply); } catch (e) {}
					const btnAbsSec = _makeToggleBtn('Abs securities', absSec);
					const btnAbsCred = _makeToggleBtn('Abs credit', absCred);

					const row1 = document.createElement('div');
					row1.className = 'tc-row';
					row1.style.width = '100%';
					row1.style.flexWrap = 'nowrap';

					const btnArea = document.createElement('button');
					btnArea.type = 'button';
					btnArea.className = 'tc-menu-btn' + (viewMode === 'area' ? ' on' : '');
					btnArea.textContent = 'Stacked Area';
					const btnBar = document.createElement('button');
					btnBar.type = 'button';
					btnBar.className = 'tc-menu-btn' + (viewMode === 'bar' ? ' on' : '');
					btnBar.textContent = 'Stacked Columns';
					function _setView(mode) {
						if (mode === viewMode) return;
						viewMode = mode;
						btnArea.className = 'tc-menu-btn' + (viewMode === 'area' ? ' on' : '');
						btnBar.className = 'tc-menu-btn' + (viewMode === 'bar' ? ' on' : '');
						_applyView(viewMode);
						setTimeout(_applyStack, 20);
					}
					btnArea.addEventListener('click', () => _setView('area'));
					btnBar.addEventListener('click', () => _setView('bar'));
					row1.appendChild(btnArea);
					row1.appendChild(btnBar);

					const row2 = document.createElement('div');
					row2.className = 'tc-row';
					row2.style.width = '100%';
					row2.appendChild(btnAbsSec);
					row2.appendChild(btnAbsCred);

					const freqWrap = _makeFreqSwitchControl(div);
					div.__tc_apply = div.__tc_apply || {};
					div.__tc_apply.syncStackMeta = _syncStackMeta;
					div.__tc_apply.stacked = () => {
						try { _syncStackMeta(); } catch (e) {}
						try { _applyView(viewMode); } catch (e) {}
						try { setTimeout(_applyStack, 10); } catch (e) {}
					};
					_addControlItems(div, [freqWrap, row1, row2].filter(Boolean));
					setTimeout(_applyStack, 60);
					return true;
				} catch (e) {
					return false;
				}
			}

			function _init3dSurfaceSelector(div) {
				try {
					if (!Array.isArray(div.data) || !div.data.length) return false;
					let surfaceIndex = -1;
					for (let i = 0; i < div.data.length; i++) {
						const t = div.data[i] || {};
						if (String(t.type || '').toLowerCase() === 'surface') { surfaceIndex = i; break; }
					}
					if (surfaceIndex < 0) return false;
					const t = div.data[surfaceIndex] || {};
					const xCur = Array.isArray(t.x) ? t.x.map(v => String(v)) : [];
					const yCur = Array.isArray(t.y) ? t.y.map(v => String(v)) : [];
					const zCur = Array.isArray(t.z) ? t.z : [];
					const meta = (div.layout && div.layout.meta && div.layout.meta.tc_surface_full) ? div.layout.meta.tc_surface_full : null;
					const x = (meta && Array.isArray(meta.x)) ? meta.x.map(v => String(v)) : xCur;
					const y = (meta && Array.isArray(meta.y)) ? meta.y.map(v => String(v)) : yCur;
					const z = (meta && Array.isArray(meta.z)) ? meta.z : zCur;
					if (x.length < 2 || y.length < 1 || !Array.isArray(z) || !Array.isArray(z[0] || [])) return false;

					div.__tc_base = div.__tc_base || {};
					if (!div.__tc_base.surface) {
						const st0 = _freqState(div);
						const bs0 = st0 ? Number(st0.blockSize || 0) : 0;
						const relIdx = (bs0 > 0) ? (surfaceIndex % bs0) : surfaceIndex;
						div.__tc_base.surface = {
							surfaceRel: relIdx,
							x: x.slice(),
							y: y.slice(),
							z: _deepCopy2d(z),
						};
					}

					const base = div.__tc_base.surface;
					const yVisible = Array.isArray(t.y) ? t.y.map(v => String(v)) : [];
					const ySelSet = new Set((yVisible || []).map(v => String(v)));
					let yChecklist = _makeChecklist(base.y, ySelSet, 34);

					function _rankRowAbsMax(z2d) {
						const scores = [];
						for (let i = 0; i < base.y.length; i++) {
							const row = (Array.isArray(z2d) && Array.isArray(z2d[i])) ? z2d[i] : [];
							let m = 0;
							for (let j = 0; j < row.length; j++) {
								const v = Number(row[j]);
								const a = Math.abs(v);
								if (a > m) m = a;
							}
							scores.push({ idx: i, score: m });
						}
						scores.sort((a, b) => (b.score - a.score));
						return scores;
					}

					const yPreset = document.createElement('select');
					yPreset.className = 'tc-menu-btn';
					const presets = [
						{ label: 'Tickers/Types: All', n: 0 },
						{ label: 'Tickers/Types: Top 10', n: 10 },
						{ label: 'Tickers/Types: Top 20', n: 20 },
						{ label: 'Tickers/Types: Top 50', n: 50 },
					];
					presets.forEach(p => {
						const opt = document.createElement('option');
						opt.value = String(p.n);
						opt.textContent = p.label;
						yPreset.appendChild(opt);
					});

					function _applyPresetSelect() {
						const n = Number(yPreset.value || '0');
						const scores = _rankRowAbsMax(base.z);
						const pickSet = new Set();
						if (!n || n <= 0 || n >= base.y.length) {
							for (const v of base.y) pickSet.add(v);
						} else {
							for (let k = 0; k < Math.min(n, scores.length); k++) {
								pickSet.add(base.y[scores[k].idx]);
							}
						}
						Array.from(yChecklist.querySelectorAll('input[type="checkbox"]')).forEach(el => {
							el.checked = pickSet.has(String(el.value));
						});
					}
					yPreset.addEventListener('change', _applyPresetSelect);

					const xStart = document.createElement('select');
					xStart.className = 'tc-menu-btn';
					const xEnd = document.createElement('select');
					xEnd.className = 'tc-menu-btn';
					for (const v of base.x) {
						const o1 = document.createElement('option');
						o1.value = v; o1.textContent = v;
						xStart.appendChild(o1);
						const o2 = document.createElement('option');
						o2.value = v; o2.textContent = v;
						xEnd.appendChild(o2);
					}
					xStart.value = base.x[0];
					xEnd.value = base.x[base.x.length - 1];

					function _syncSurfaceMeta() {
						try {
							const m2 = (div.layout && div.layout.meta && div.layout.meta.tc_surface_full) ? div.layout.meta.tc_surface_full : null;
							if (!m2) return;
							base.x = (Array.isArray(m2.x) ? m2.x.map(v => String(v)) : base.x);
							base.y = (Array.isArray(m2.y) ? m2.y.map(v => String(v)) : base.y);
							base.z = (Array.isArray(m2.z) ? _deepCopy2d(m2.z) : base.z);

							const prev0 = String(xStart.value || '');
							const prev1 = String(xEnd.value || '');
							xStart.innerHTML = '';
							xEnd.innerHTML = '';
							for (const v of base.x) {
								const o1 = document.createElement('option');
								o1.value = v; o1.textContent = v;
								xStart.appendChild(o1);
								const o2 = document.createElement('option');
								o2.value = v; o2.textContent = v;
								xEnd.appendChild(o2);
							}
							if (prev0 && base.x.indexOf(prev0) >= 0) xStart.value = prev0; else xStart.value = base.x[0];
							if (prev1 && base.x.indexOf(prev1) >= 0) xEnd.value = prev1; else xEnd.value = base.x[base.x.length - 1];

							const pickedY = _checkedValues(yChecklist);
							const pickSet = new Set(pickedY.map(v => String(v)));
							const newChecklist = _makeChecklist(base.y, pickSet, 34);
							try {
								if (yChecklist && yChecklist.parentNode) yChecklist.parentNode.replaceChild(newChecklist, yChecklist);
							} catch (e) {}
							yChecklist = newChecklist;
						} catch (e) {}
					}

					function _apply3d() {
						const pickedY = _checkedValues(yChecklist);
						const ySet = new Set(pickedY);

						let x2 = [];
						let colIdx = [];
						let i0 = base.x.indexOf(String(xStart.value));
						let i1 = base.x.indexOf(String(xEnd.value));
						if (i0 < 0) i0 = 0;
						if (i1 < 0) i1 = base.x.length - 1;
						if (i0 > i1) { const tmp = i0; i0 = i1; i1 = tmp; }
						x2 = base.x.slice(i0, i1 + 1);
						for (let j = i0; j <= i1; j++) colIdx.push(j);

						const y2 = [];
						const rowIdx = [];
						for (let i = 0; i < base.y.length; i++) {
							if (pickedY.length === 0 || ySet.has(base.y[i])) { y2.push(base.y[i]); rowIdx.push(i); }
						}

						const z2 = [];
						for (const ri of rowIdx) {
							const srcRow = base.z[ri] || [];
							const row = [];
							for (const cj of colIdx) row.push(Number(srcRow[cj]));
							z2.push(row);
						}

						let meanIdx = -1;
						let medIdx = -1;
						try {
							const st = _freqState(div);
							const bs = st ? Number(st.blockSize || 0) : 0;
							const off = _freqOffset(div);
							const sIdx = off + Number(base.surfaceRel || 0);
							const end = (bs > 0) ? Math.min((div.data || []).length, off + bs) : (div.data || []).length;
							for (let i = Math.max(0, off); i < end; i++) {
								if (i === sIdx) continue;
								const tt = div.data[i] || {};
								if (String(tt.type || '').toLowerCase() !== 'surface') continue;
								const nm = String(tt.name || '').toLowerCase();
								if (nm.includes('mean')) meanIdx = i;
								if (nm.includes('median')) medIdx = i;
							}
						} catch (e) {}
						let zMean = null;
						let zMed = null;
						try {
							if (z2.length && z2[0] && z2[0].length) {
								const cols = z2[0].length;
								const meanRow = [];
								const medRow = [];
								for (let j = 0; j < cols; j++) {
									const col = [];
									for (let i = 0; i < z2.length; i++) col.push(Number(z2[i][j]));
									const n = col.length;
									const m = col.reduce((a, b) => a + b, 0) / Math.max(1, n);
									const s = col.slice().sort((a, b) => a - b);
									const md = (n % 2) ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
									meanRow.push(m);
									medRow.push(md);
								}
								zMean = Array.from({length: z2.length}, _ => meanRow.slice());
								zMed = Array.from({length: z2.length}, _ => medRow.slice());
							}
						} catch (e) {}

						const layout = (div && div.layout) ? div.layout : {};
						const sceneKeys = Object.keys(layout).filter(k => k.startsWith('scene'));
						const sk = sceneKeys.length ? sceneKeys.sort()[0] : 'scene';
						const xStride = Math.max(1, Math.floor(x2.length / 12));
						const xTicks = x2.filter((_, i) => (i % xStride) === 0);
						const yText = y2.map(v => _shortenLabel(v, 28));

						try {
							const off = _freqOffset(div);
							const sIdx = off + Number(base.surfaceRel || 0);
							const idxs = [sIdx];
							const xArr = [x2];
							const yArr = [y2];
							const zArr = [z2];
							if (meanIdx >= 0 && zMean) { idxs.push(meanIdx); xArr.push(x2); yArr.push(y2); zArr.push(zMean); }
							if (medIdx >= 0 && zMed) { idxs.push(medIdx); xArr.push(x2); yArr.push(y2); zArr.push(zMed); }
							window.Plotly.restyle(div, { 'x': xArr, 'y': yArr, 'z': zArr }, idxs);
							const rel = {};
							rel[`${sk}.xaxis.tickmode`] = 'array';
							rel[`${sk}.xaxis.tickvals`] = xTicks;
							rel[`${sk}.xaxis.ticktext`] = xTicks;
							rel[`${sk}.yaxis.tickmode`] = 'array';
							rel[`${sk}.yaxis.tickvals`] = y2;
							rel[`${sk}.yaxis.ticktext`] = yText;
							window.Plotly.relayout(div, rel);
							window.Plotly.Plots.resize(div);
						} catch (e) {}
					}

					const applyBtn = _makeButton('Apply', _apply3d);
					applyBtn.className = 'tc-menu-btn';
					const resetBtn = _makeButton('Reset', () => {
						_setAllChecks(yChecklist, false);
						yPreset.value = '0';
						xStart.value = base.x[0];
						xEnd.value = base.x[base.x.length - 1];
						_apply3d();
					});
					resetBtn.className = 'tc-menu-btn';

					const allBtn = document.createElement('button');
					allBtn.type = 'button';
					allBtn.className = 'tc-menu-btn';
					allBtn.textContent = 'All';
					allBtn.addEventListener('click', () => { _setAllChecks(yChecklist, true); });
					const noneBtn = document.createElement('button');
					noneBtn.type = 'button';
					noneBtn.className = 'tc-menu-btn';
					noneBtn.textContent = 'None';
					noneBtn.addEventListener('click', () => { _setAllChecks(yChecklist, false); });

					const pickPopup = document.createElement('div');
					pickPopup.style.display = 'grid';
					pickPopup.style.gap = '10px';
					const presetRow = document.createElement('div');
					presetRow.className = 'tc-row';
					const prLab = document.createElement('span');
					prLab.className = 'tc-mini';
					prLab.textContent = 'Preset:';
					prLab.style.marginRight = '6px';
					presetRow.appendChild(prLab);
					presetRow.appendChild(yPreset);
					pickPopup.appendChild(presetRow);
					pickPopup.appendChild(yChecklist);
					const row = document.createElement('div');
					row.className = 'tc-row';
					row.appendChild(allBtn);
					row.appendChild(noneBtn);
					pickPopup.appendChild(row);
					const pickMenu = _makeMenu('Pick tickers/types', pickPopup);

					const xWrap = document.createElement('div');
					xWrap.className = 'tc-menu';
					const xl = document.createElement('span');
					xl.className = 'tc-mini';
					xl.textContent = 'X:';
					xl.style.marginRight = '6px';
					xWrap.appendChild(xl);
					xWrap.appendChild(xStart);
					const dash = document.createElement('span');
					dash.className = 'tc-mini';
					dash.textContent = '–';
					dash.style.margin = '0 6px';
					xWrap.appendChild(dash);
					xWrap.appendChild(xEnd);

					const freqWrap = _makeFreqSwitchControl(div);
					div.__tc_apply = div.__tc_apply || {};
					div.__tc_apply.syncSurfaceMeta = _syncSurfaceMeta;
					div.__tc_apply.surfaceApply = () => { try { _syncSurfaceMeta(); } catch (e) {} try { _apply3d(); } catch (e) {} };
					_add3dChoiceFieldControls(div, [freqWrap, pickMenu.wrap, xWrap, applyBtn, resetBtn].filter(Boolean));
					return true;
				} catch (e) {
					return false;
				}
			}

			function _initControlsOnce() {
				if (!window.Plotly) return;
				const bar = document.getElementById('tc-controls');
				if (bar && bar.classList && bar.classList.contains('has-controls')) return;
				const divs = Array.from(document.querySelectorAll('.plotly-graph-div'));
				if (!divs.length) return;
				for (const div of divs) {
					if (div.__tc_controls_inited) continue;
					const added = _init3dSurfaceSelector(div) || _initStackedSelector(div) || _initDistributionBarSelector(div) || _initFinalPnlBarSelector(div) || _initViewSwitchSelector(div) || _initTopNSelector(div) || _initFreqSelector(div);
					if (added) {
						div.__tc_controls_inited = true;
						break;
					}
				}
			}

			function _themeUpdate(dark, div) {
				const bg = dark ? '#0b1220' : '#ffffff';
				const paper = bg;
				const font = dark ? '#e5e7eb' : '#111827';
				const mutedGrid = dark ? 'rgba(229,231,235,0.16)' : 'rgba(17,24,39,0.12)';
				const zero = dark ? 'rgba(229,231,235,0.24)' : 'rgba(17,24,39,0.22)';
				const ctrlBg = dark ? 'rgba(23,23,24,0.55)' : 'rgba(247,247,247,0.55)';
				const ctrlBorder = dark ? 'rgba(255,255,255,0.22)' : 'rgba(0,0,0,0.18)';
				const ctrlFont = dark ? '#f0f0f0' : '#111827';

				const hasLegend = _hasLegend(div);
				const hasCb = _hasColorbar(div);
				const uiHost = document.getElementById('tc-ui');
				const hasPanels = false;
				const u = {
					'template': dark ? 'plotly_dark' : 'plotly_white',
					'paper_bgcolor': paper,
					'plot_bgcolor': bg,
					'font.color': font,
					'showlegend': hasLegend,
					'legend.bgcolor': 'rgba(0,0,0,0)',
					'legend.orientation': 'v',
					'legend.x': hasLegend ? (hasCb ? 0.02 : 1.01) : 0,
					'legend.xanchor': hasLegend ? 'left' : 'left',
					'legend.y': hasCb ? 0.92 : 1.0,
					'legend.yanchor': 'top',
					'margin.l': 48,
					'margin.r': (hasLegend && !hasPanels) ? 200 : 48,
					'margin.t': 118,
					'margin.b': 56,
					'title.x': 0.5,
					'title.xanchor': 'center',
					'title.y': 0.985,
					'title.yanchor': 'top',
					'hoverlabel.bgcolor': dark ? 'rgba(17,24,39,0.92)' : 'rgba(255,255,255,0.96)',
					'hoverlabel.font.color': dark ? '#f9fafb' : '#111827',
					'hoverlabel.bordercolor': dark ? 'rgba(229,231,235,0.35)' : 'rgba(17,24,39,0.25)',
				};

				try {
					const layout = (div && div.layout) ? div.layout : {};
					const keys = Object.keys(layout);
					const axisKeys = keys.filter(k => k.startsWith('xaxis') || k.startsWith('yaxis'));
					axisKeys.forEach(k => {
						u[`${k}.gridcolor`] = mutedGrid;
						u[`${k}.zerolinecolor`] = zero;
						u[`${k}.linecolor`] = zero;
						u[`${k}.tickcolor`] = zero;
						u[`${k}.color`] = font;
						u[`${k}.title.font.color`] = font;
					});

					const sceneKeys = keys.filter(k => k.startsWith('scene'));
					sceneKeys.forEach(sk => {
						u[`${sk}.bgcolor`] = bg;
						['xaxis','yaxis','zaxis'].forEach(ax => {
							u[`${sk}.${ax}.color`] = font;
							u[`${sk}.${ax}.gridcolor`] = mutedGrid;
							u[`${sk}.${ax}.zerolinecolor`] = zero;
							u[`${sk}.${ax}.backgroundcolor`] = bg;
							u[`${sk}.${ax}.showbackground`] = true;
						});
					});

					const ums = Array.isArray(layout.updatemenus) ? layout.updatemenus : [];
					const baseY = 1.22;
					const step = 0.095;
					ums.forEach((_, i) => {
						u[`updatemenus[${i}].x`] = 0.0;
						u[`updatemenus[${i}].xanchor`] = 'left';
						u[`updatemenus[${i}].y`] = baseY - i * step;
						u[`updatemenus[${i}].yanchor`] = 'top';
						u[`updatemenus[${i}].bgcolor`] = ctrlBg;
						u[`updatemenus[${i}].bordercolor`] = ctrlBorder;
						u[`updatemenus[${i}].borderwidth`] = 1;
						u[`updatemenus[${i}].font.color`] = ctrlFont;
						u[`updatemenus[${i}].font.size`] = 12;
					});
				} catch (e) { /* ignore */ }
				return u;
			}

			function applyTheme(dark) {
				document.body.classList.toggle('dark', !!dark);
				if (!window.Plotly) return;
				const divs = Array.from(document.querySelectorAll('.plotly-graph-div'));
				divs.forEach(div => {
					try { window.Plotly.relayout(div, _themeUpdate(!!dark, div)); } catch (e) {}
					try { _applyTraceTheme(!!dark, div); } catch (e) {}
					try { window.Plotly.relayout(div, { 'title.text': '' }); } catch (e) {}
					try { window.Plotly.Plots.resize(div); } catch (e) {}
				});
			}

			function _fitToViewport() {
				try {
					const plotWrap = document.getElementById('tc-plot');
					if (!plotWrap) return;
					const divs = Array.from(plotWrap.querySelectorAll('.plotly-graph-div'));
					if (!divs.length || !window.Plotly) return;
					const h = plotWrap.clientHeight;
					const w = plotWrap.clientWidth;
					if (!h || h < 180) return;
					if (!w || w < 240) return;
					divs.forEach(div => {
						try { div.style.height = h + 'px'; } catch (e) {}
						try { div.style.width = w + 'px'; } catch (e) {}
						try { window.Plotly.relayout(div, { height: h, width: w }); } catch (e) {}
						try { window.Plotly.Plots.resize(div); } catch (e) {}
					});
				} catch (e) {}
			}

			window.addEventListener('message', (ev) => {
				if (!ev || !ev.data) return;
				if (typeof ev.data.tc_theme === 'boolean') applyTheme(ev.data.tc_theme);
				if (ev.data.tc_resize) {
					setTimeout(_fitToViewport, 40);
					setTimeout(_fitToViewport, 260);
					setTimeout(_fitToViewport, 560);
					try {
						const divs = Array.from(document.querySelectorAll('.plotly-graph-div'));
						divs.forEach(d => { try { window.Plotly && window.Plotly.Plots && window.Plotly.Plots.resize(d); } catch(e) {} });
					} catch (e) {}
				}
				if (ev.data && ev.data.tc_export) {
					_handleExportMessage(ev.data.tc_export);
				}
			});

			document.addEventListener('DOMContentLoaded', () => {
				applyTheme(false);
				setTimeout(_initControlsOnce, 90);
				setTimeout(_initControlsOnce, 520);
				try {
					window.parent && window.parent.postMessage({ tc_export_caps: _exportCapsPlotly() }, '*');
				} catch (e) {}
				setTimeout(() => applyTheme(document.body.classList.contains('dark')), 60);
				setTimeout(() => applyTheme(document.body.classList.contains('dark')), 450);
				setTimeout(_fitToViewport, 120);
				setTimeout(_fitToViewport, 600);
				window.addEventListener('resize', () => setTimeout(_fitToViewport, 50));
				try {
					const plotWrap = document.getElementById('tc-plot');
					if (plotWrap && window.ResizeObserver) {
						const ro = new ResizeObserver(() => setTimeout(_fitToViewport, 30));
						ro.observe(plotWrap);
					}
				} catch (e) {}
			});
		})();
	</script>
</body>
</html>"""
	return tpl.replace("__TITLE__", safe_title).replace("__FRAGMENT__", fragment)


def _visual_to_full_html(results: List[Any], spec: VisualSpec, board: Optional[str]) -> str:
	try:
		out = spec.builder(results, board)
		if isinstance(out, go.Figure):
			return _figure_to_html(out)
		if isinstance(out, str):
			return out
		raise TypeError(f"Unsupported visual output type: {type(out)}")
	except Exception as e:
		msg = str(e)
		return f"""<!doctype html>
<html><head><meta charset='utf-8'/><title>Error</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;padding:16px;}}pre{{white-space:pre-wrap;}}</style>
</head><body>
<h3>Failed to build: {spec.label}</h3>
<pre>{msg}</pre>
</body></html>"""


def build_visuals_dashboard_html(results: List[Any], specs: List[VisualSpec], board: Optional[str] = None) -> str:

	def _category(key: str) -> str:
		k = (key or "").lower()
		if k.startswith("overview") or k in ("pct_delta", "waterfall_candle", "portfolio_value"):
			return "Overview"
		if k.startswith("stack_"):
			return "Portfolio Composition"
		if "tickers" in k or k.endswith("_tickers"):
			return "Tickers"
		if k.startswith("types") or "_types" in k or k.endswith("_types"):
			return "Types"
		if k.startswith("dist_"):
			return "Distributions"
		return "Other"

	category_order = ["Overview", "Portfolio Composition", "Tickers", "Types", "Distributions", "Other"]
	cat_rank = {c: i for i, c in enumerate(category_order)}

	items: List[Dict[str, str]] = []
	for s in specs:
		html = _visual_to_full_html(results, s, board)
		b64 = base64.b64encode(html.encode("utf-8")).decode("ascii")
		items.append({"key": s.key, "label": s.label, "category": _category(s.key), "b64": b64})

	items.sort(key=lambda d: (cat_rank.get(d.get("category", "Other"), 999), d.get("label", "")))

	payload = json.dumps(items, ensure_ascii=False)
	title = _safe_title(board, "Trade Calculator — ALL Visuals")
	category_order_json = json.dumps(category_order, ensure_ascii=False)

	return f"""<!doctype html>
<html>
<head>
	<meta charset='utf-8'/>
	<meta name='viewport' content='width=device-width, initial-scale=1'/>
	<title>{title}</title>
	<style>
		:root {{
			--bg:#ffffff; --fg:#111; --muted:#666; --border:#ddd;
			--panel:#f7f7f7; --panel2:#f1f1f1; --hover:#ececec;
			--active:#dfe7ff; --accent:#2f6feb;
		}}
		body.dark {{
			--bg:#0f0f10; --fg:#f0f0f0; --muted:#aaa; --border:#2e2e2f;
			--panel:#171718; --panel2:#1e1e1f; --hover:#242425;
			--active:#1f2a4a; --accent:#7aa2ff;
		}}
		* {{ box-sizing: border-box; }}
		body {{ margin:0; font-family: Segoe UI, Arial, sans-serif; background:var(--bg); color:var(--fg); }}
		.app {{ display:flex; height:100vh; width:100vw; overflow:hidden; }}

		.sidebar {{ width: 320px; min-width: 260px; background:var(--panel); border-right:1px solid var(--border); display:flex; flex-direction:column; transition: width 0.18s ease; }}
		.sidebar.collapsed {{ width: 56px; min-width: 56px; }}
		.side-top {{ display:flex; align-items:center; gap:10px; padding:10px 10px; border-bottom:1px solid var(--border); }}
		.hamb {{ width:36px; height:36px; display:flex; align-items:center; justify-content:center; border:1px solid var(--border); border-radius:10px; background:var(--panel2); cursor:pointer; user-select:none; }}
		.side-title {{ font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
		.sidebar.collapsed .side-title {{ display:none; }}

		.side-actions {{ display:flex; gap:8px; padding:10px 10px; border-bottom:1px solid var(--border); }}
		.sidebar.collapsed .side-actions {{ display:none; }}
		.btn {{ padding:6px 10px; border:1px solid var(--border); background:var(--panel2); color:var(--fg); cursor:pointer; border-radius:10px; }}
		.note {{ color:var(--muted); font-size:12px; padding:10px 10px; border-bottom:1px solid var(--border); }}
		.sidebar.collapsed .note {{ display:none; }}

		.nav {{ padding:8px 6px 10px 6px; overflow:auto; position:relative; }}
		.sidebar.collapsed .nav {{ padding:8px 4px; }}

		.cat {{ margin:6px 4px; border:1px solid var(--border); border-radius:12px; overflow:visible; background:var(--panel2); position:relative; }}
		.cat-head {{ display:flex; align-items:center; gap:10px; padding:10px 10px; cursor:pointer; user-select:none; position:sticky; top:0; z-index:3; background:var(--panel2); }}
		.chev {{ width:18px; text-align:center; color:var(--muted); }}
		.cat-name {{ font-weight:600; flex:1; }}
		.cat-count {{ color:var(--muted); font-size:12px; }}
		.sidebar.collapsed .cat-name, .sidebar.collapsed .cat-count {{ display:none; }}

		.cat-items {{ display:none; padding:6px 6px 10px 34px; background:var(--panel); }}
		.cat.open .cat-items {{ display:block; }}
		.sidebar.collapsed .cat-items {{ display:none !important; }}

		.item {{ padding:8px 10px; border-radius:10px; cursor:pointer; user-select:none; margin:4px 0; }}
		.item:hover {{ background:var(--hover); }}
		.item.active {{ background:var(--active); outline: 1px solid rgba(47,111,235,0.35); }}
		.item-key {{ color:var(--muted); font-size:11px; }}

		.main {{ flex:1; display:flex; flex-direction:column; min-width:0; }}
		.main-top {{ display:flex; align-items:center; gap:12px; padding:10px 12px; border-bottom:1px solid var(--border); }}
		.main-title {{ font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
		.main-sub {{ color:var(--muted); font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
		.spacer {{ flex:1; }}
		.export-wrap {{ position: relative; display:flex; align-items:center; }}
		.export-menu {{
			position:absolute;
			right: 0;
			top: calc(100% + 8px);
			min-width: 260px;
			max-width: min(420px, calc(100vw - 40px));
			background: var(--panel);
			border: 1px solid var(--border);
			border-radius: 12px;
			box-shadow: 0 14px 40px rgba(0,0,0,0.18);
			padding: 8px;
			display:none;
			z-index: 100;
		}}
		body.dark .export-menu {{ box-shadow: 0 14px 40px rgba(0,0,0,0.55); }}
		.export-menu.open {{ display:block; }}
		.export-section {{ padding: 6px 6px 2px 6px; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }}
		.export-item {{ width: 100%; text-align:left; padding: 8px 10px; border-radius: 10px; border: 1px solid transparent; background: transparent; color: var(--fg); cursor: pointer; }}
		.export-item:hover {{ background: var(--hover); }}
		.export-item:disabled {{ opacity: 0.55; cursor: not-allowed; }}
		.export-sep {{ height: 1px; background: var(--border); margin: 6px 2px; }}
		iframe {{ width:100%; height:100%; border:0; background:var(--bg); }}
		.frame-wrap {{ flex:1; min-height:0; }}
	</style>
</head>
<body>
	<div class='app'>
		<aside id='sidebar' class='sidebar'>
			<div class='side-top'>
				<div class='hamb' title='Collapse/Expand' onclick='toggleSidebar()'>☰</div>
				<div class='side-title' title='{title}'>{title}</div>
			</div>
			<div class='side-actions'>
				<button class='btn' onclick='setTheme(false)'>Light</button>
				<button class='btn' onclick='setTheme(true)'>Dark</button>
			</div>
			<div class='note'>Click a category to expand, then click a visual.</div>
			<div id='nav' class='nav'></div>
		</aside>

		<main class='main'>
			<div class='main-top'>
				<div class='main-title' id='currentTitle'>Loading…</div>
				<div class='main-sub' id='currentKey'></div>
				<div class='spacer'></div>
				<div class='export-wrap'>
					<button class='btn' id='exportBtn' title='Export'>Export ▾</button>
					<div class='export-menu' id='exportMenu'></div>
				</div>
			</div>
			<div class='frame-wrap'>
				<iframe id='frame' sandbox='allow-scripts allow-same-origin allow-forms allow-popups allow-downloads'></iframe>
			</div>
		</main>
	</div>

	<script>
		const VIS = {payload};
		const CATEGORY_ORDER = {category_order_json};
		const nav = document.getElementById('nav');
		const frame = document.getElementById('frame');
		const currentTitle = document.getElementById('currentTitle');
		const currentKey = document.getElementById('currentKey');
		const sidebar = document.getElementById('sidebar');
		const exportBtn = document.getElementById('exportBtn');
		const exportMenu = document.getElementById('exportMenu');
		let _currentIdx = -1;
		let _exportCaps = null;
		let _exportKind = '';

		function b64ToUtf8(b64) {{
			const bin = atob(b64);
			const bytes = Uint8Array.from(bin, c => c.charCodeAt(0));
			return new TextDecoder('utf-8').decode(bytes);
		}}

		function _downloadBlob(filename, mime, text) {{
			try {{
				const blob = new Blob([text], {{ type: mime || 'application/octet-stream' }});
				const url = URL.createObjectURL(blob);
				const a = document.createElement('a');
				a.href = url;
				a.download = filename || 'export';
				a.style.display = 'none';
				document.body.appendChild(a);
				a.click();
				setTimeout(() => {{
					try {{ URL.revokeObjectURL(url); }} catch (e) {{}}
					try {{ a.remove(); }} catch (e) {{}}
				}}, 120);
				return true;
			}} catch (e) {{
				return false;
			}}
		}}

		function _exportCurrentHtml() {{
			try {{
				if (_currentIdx == null || _currentIdx < 0) return;
				const v = VIS[_currentIdx];
				if (!v || !v.b64) return;
				const html = b64ToUtf8(v.b64);
				const fn = (currentTitle && currentTitle.textContent) ? currentTitle.textContent : 'export';
				_downloadBlob(fn + '.html', 'text/html;charset=utf-8', html);
			}} catch (e) {{}}
		}}

		function _postThemeToFrame() {{
			try {{
				const dark = document.body.classList.contains('dark');
				if (frame && frame.contentWindow) frame.contentWindow.postMessage({{tc_theme: dark}}, '*');
			}} catch (e) {{}}
		}}

		function _postResizeToFrame() {{
			try {{
				if (frame && frame.contentWindow) frame.contentWindow.postMessage({{tc_resize: true}}, '*');
			}} catch (e) {{}}
		}}

		function _postExportToFrame(payload) {{
			try {{
				if (frame && frame.contentWindow) frame.contentWindow.postMessage({{ tc_export: payload }}, '*');
			}} catch (e) {{}}
		}}

		function _requestExportCaps() {{
			_exportCaps = null;
			_exportKind = '';
			renderExportMenu(null);
			_postExportToFrame({{ action: 'caps' }});
			setTimeout(() => {{
				try {{
					if (_exportCaps != null) return;
					_exportCaps = {{ kind: '', supported: false, image: [], data: [], other: ['html'], clipboard: [] }};
					renderExportMenu(_exportCaps);
				}} catch (e) {{}}
			}}, 900);
		}}

		function _closeExportMenu() {{
			try {{ exportMenu.classList.remove('open'); }} catch (e) {{}}
		}}

		function _toggleExportMenu() {{
			const isOpen = exportMenu.classList.contains('open');
			if (isOpen) _closeExportMenu();
			else exportMenu.classList.add('open');
		}}

		function _addExportSection(title) {{
			const d = document.createElement('div');
			d.className = 'export-section';
			d.textContent = title;
			exportMenu.appendChild(d);
		}}

		function _addExportItem(label, payload, enabled) {{
			const b = document.createElement('button');
			b.type = 'button';
			b.className = 'export-item';
			b.textContent = label;
			b.disabled = !enabled;
			b.onclick = () => {{
				_closeExportMenu();
				if (payload && payload._local === 'html') {{
					_exportCurrentHtml();
					return;
				}}
				const fn = (currentTitle && currentTitle.textContent) ? currentTitle.textContent : 'export';
				_postExportToFrame({{ ...payload, action: 'run', filename: fn, kind: _exportKind }});
			}};
			exportMenu.appendChild(b);
		}}

		function _addSep() {{
			const s = document.createElement('div');
			s.className = 'export-sep';
			exportMenu.appendChild(s);
		}}

		function renderExportMenu(caps) {{
			exportMenu.innerHTML = '';
			const supported = !!(caps && caps.supported);
			const kind = caps && caps.kind ? String(caps.kind) : '';
			const img = (caps && Array.isArray(caps.image)) ? caps.image : [];
			const data = (caps && Array.isArray(caps.data)) ? caps.data : [];
			const other = (caps && Array.isArray(caps.other)) ? caps.other : [];
			const clip = (caps && Array.isArray(caps.clipboard)) ? caps.clipboard : [];

			if (!caps) {{
				_addExportSection('Export');
				_addExportItem('Detecting export options…', {{ scope: 'noop' }}, false);
				return;
			}}

			if (!supported) {{
				_addExportSection('Export');
				_addExportItem('No export available for this visual', {{ scope: 'noop' }}, false);
				_addSep();
				_addExportSection('Page');
				_addExportItem('Download HTML (page)', {{ _local: 'html' }}, true);
				return;
			}}

			if (img.length) {{
				_addExportSection('Image');
				img.forEach(fmt => _addExportItem(('Download ' + String(fmt).toUpperCase()), {{ scope: 'image', format: String(fmt) }}, true));
				_addSep();
			}}

			_addExportSection('Data');
			if (data.includes('csv')) _addExportItem('Download CSV (long format)', {{ scope: 'data', format: 'csv' }}, true);
			if (data.includes('json')) _addExportItem('Download JSON (data + layout)', {{ scope: 'data', format: 'json' }}, true);
			if (data.includes('xlsx')) _addExportItem('Download XLSX', {{ scope: 'data', format: 'xlsx' }}, true);
			if (data.includes('pdf')) _addExportItem('Download PDF', {{ scope: 'data', format: 'pdf' }}, true);
			if (data.includes('html')) _addExportItem('Download HTML (table)', {{ scope: 'data', format: 'html' }}, true);

			if (clip.length) {{
				_addSep();
				_addExportSection('Clipboard');
				if (clip.includes('csv')) _addExportItem('Copy CSV', {{ scope: 'clipboard', format: 'csv' }}, true);
				if (clip.includes('json')) _addExportItem('Copy JSON', {{ scope: 'clipboard', format: 'json' }}, true);
			}}

			if (other.length) {{
				_addSep();
				_addExportSection('Page');
				if (other.includes('html')) _addExportItem('Download HTML (page)', {{ _local: 'html' }}, true);
			}}
		}}

		window.addEventListener('message', (ev) => {{
			try {{
				if (!ev || !ev.data) return;
				if (ev.data.tc_export_caps) {{
					_exportCaps = ev.data.tc_export_caps;
					_exportKind = _exportCaps && _exportCaps.kind ? String(_exportCaps.kind) : '';
					renderExportMenu(_exportCaps);
				}}
			}} catch (e) {{}}
		}});

		exportBtn.addEventListener('click', (ev) => {{
			try {{ ev && ev.stopPropagation && ev.stopPropagation(); }} catch (e) {{}}
			_toggleExportMenu();
		}});
		document.addEventListener('click', (ev) => {{
			const t = ev && ev.target;
			if (!t) return;
			if (exportMenu.contains(t) || exportBtn.contains(t)) return;
			_closeExportMenu();
		}}, true);

		function setTheme(dark) {{
			document.body.classList.toggle('dark', !!dark);
			localStorage.setItem('tc_theme', dark ? 'dark' : 'light');
			_postThemeToFrame();
		}}

		function toggleSidebar() {{
			const collapsed = sidebar.classList.toggle('collapsed');
			localStorage.setItem('tc_sidebar', collapsed ? 'collapsed' : 'open');
			_postResizeToFrame();
			setTimeout(_postResizeToFrame, 230);
			setTimeout(_postResizeToFrame, 520);
		}}

		function selectVisual(idx) {{
			_currentIdx = idx;
			const v = VIS[idx];
			frame.onload = () => {{ _postThemeToFrame(); _postResizeToFrame(); _requestExportCaps(); }};
			frame.srcdoc = b64ToUtf8(v.b64);
			currentTitle.textContent = v.label;
			currentKey.textContent = v.key;
			document.title = v.label;
			document.querySelectorAll('.item').forEach(el => el.classList.remove('active'));
			const el = document.querySelector(`.item[data-idx='${{idx}}']`);
			if (el) el.classList.add('active');
		}}

		function toggleCategory(catEl) {{
			catEl.classList.toggle('open');
			const chev = catEl.querySelector('.chev');
			if (chev) chev.textContent = catEl.classList.contains('open') ? '▾' : '▸';
		}}

		const grouped = new Map();
		VIS.forEach((v, idx) => {{
			const c = v.category || 'Other';
			if (!grouped.has(c)) grouped.set(c, []);
			grouped.get(c).push({{...v, idx}});
		}});

		function catRank(name) {{
			const i = CATEGORY_ORDER.indexOf(name);
			return i === -1 ? 999 : i;
		}}

		const cats = Array.from(grouped.keys()).sort((a,b) => catRank(a) - catRank(b) || a.localeCompare(b));
		cats.forEach((catName, catIdx) => {{
			const cat = document.createElement('div');
			cat.className = (catIdx === 0) ? 'cat open' : 'cat';

			const head = document.createElement('div');
			head.className = 'cat-head';
			head.onclick = () => toggleCategory(cat);

			const chev = document.createElement('div');
			chev.className = 'chev';
			chev.textContent = cat.classList.contains('open') ? '▾' : '▸';
			head.appendChild(chev);

			const name = document.createElement('div');
			name.className = 'cat-name';
			name.textContent = catName;
			head.appendChild(name);

			const count = document.createElement('div');
			count.className = 'cat-count';
			count.textContent = String(grouped.get(catName).length);
			head.appendChild(count);

			const items = document.createElement('div');
			items.className = 'cat-items';
			grouped.get(catName).forEach(v => {{
				const it = document.createElement('div');
				it.className = 'item';
				it.dataset.idx = String(v.idx);
				it.title = v.key;
				it.onclick = () => selectVisual(v.idx);
				it.innerHTML = `<div>${{v.label}}</div><div class='item-key'>${{v.key}}</div>`;
				items.appendChild(it);
			}});

			cat.appendChild(head);
			cat.appendChild(items);
			nav.appendChild(cat);
		}});

		const savedTheme = localStorage.getItem('tc_theme');
		if (savedTheme === 'dark') setTheme(true);
		const savedSide = localStorage.getItem('tc_sidebar');
		if (savedSide === 'collapsed') sidebar.classList.add('collapsed');

		if (VIS.length) {{
			const DEFAULT_KEY = 'portfolio_value';
			let di = 0;
			for (let i = 0; i < VIS.length; i++) {{ if (VIS[i].key === DEFAULT_KEY) {{ di = i; break; }} }}
			selectVisual(di);
		}}
	</script>
</body>
</html>"""


def open_visuals_dashboard(results: List[Any], specs: List[VisualSpec], board: Optional[str] = None) -> None:
	if not specs:
		return
	html = build_visuals_dashboard_html(results, specs, board)
	show_html_in_browser(html, title=_safe_title(board, "ALL Visuals"))


def show_html_in_browser(html: str, *, title: str) -> None:
	try:
		payload = html.encode("utf-8")

		class _Handler(BaseHTTPRequestHandler):
			def do_GET(self):
				if self.path not in ("/", "/index.html"):
					self.send_response(404)
					self.end_headers()
					return
				self.send_response(200)
				self.send_header("Content-Type", "text/html; charset=utf-8")
				self.send_header("Content-Length", str(len(payload)))
				self.end_headers()
				self.wfile.write(payload)
				threading.Thread(target=self.server.shutdown, daemon=True).start()

			def log_message(self, format, *args):
				return

		httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
		port = httpd.server_address[1]
		threading.Thread(target=httpd.serve_forever, daemon=True).start()
		webbrowser.open(f"http://127.0.0.1:{port}/", new=2, autoraise=True)
		return
	except Exception:

		f = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8")
		try:
			f.write(html)
			f.flush()
			tmp_path = f.name
		finally:
			try:
				f.close()
			except Exception:
				pass
		_TEMP_FILES.append(tmp_path)
		webbrowser.open("file://" + tmp_path, new=2, autoraise=True)

		def _del_later(path: str) -> None:
			import os
			time.sleep(10)
			try:
				os.remove(path)
			except Exception:
				pass
		threading.Thread(target=_del_later, args=(tmp_path,), daemon=True).start()


def _add_theme_toggle(fig: go.Figure, *, x: float = 1.0, y: float = 1.12) -> None:
	return


def validate_results(results: Any) -> Tuple[bool, str]:
	if not isinstance(results, (list, tuple)):
		return False, "results is not a list/tuple"
	if len(results) < 10:
		return False, f"results too short: {len(results)}"
	return True, "ok"


def available_visuals(results: List[Any]) -> List[VisualSpec]:
	ok, _ = validate_results(results)
	if not ok:
		return []

	results_d = _results_view(results, 0)


	lookup_days = results_d[1] if len(results_d) > 1 else None
	dt_ok = lookup_days is not None and isinstance(lookup_days, (list, tuple, np.ndarray)) and len(lookup_days) > 0

	tickers = results_d[2] if len(results_d) > 2 else None
	types = results_d[3] if len(results_d) > 3 else None
	tickers_profit = results_d[0] if len(results_d) > 0 else None
	ticker_volume_prices = results_d[4] if len(results_d) > 4 else None
	start_money = results_d[13] if len(results_d) > 13 else None

	specs: List[VisualSpec] = []
	if dt_ok and results_d[9] is not None:
		specs.append(VisualSpec("portfolio_value", "Portfolio Value (Line / Candle / Waterfall)", _wrap_visual_builder(build_portfolio_value_all_figure, kind="figure")))
		specs.append(VisualSpec("pct_delta", "% Portfolio Value Δ", _wrap_visual_builder(build_portfolio_value_pct_delta_figure, kind="figure")))
		specs.append(VisualSpec("overview_flows", "Profit / Cash / Credit (Lines)", _wrap_visual_builder(build_overview_flows_figure, kind="figure")))



	if dt_ok and results_d[7] is not None and results_d[8] is not None:
		if ticker_volume_prices is not None and tickers is not None and types is not None:
			specs.append(VisualSpec("stack_tickers", "Portfolio Value — Stacked (Tickers excl. futures + Cash + Credit)", _wrap_visual_builder(build_portfolio_value_stacked_by_ticker_figure, kind="figure")))
			specs.append(VisualSpec("stack_tickers_pct", "% Portfolio Value — 100% Stacked (Tickers excl. futures + Cash + Credit)", _wrap_visual_builder(build_portfolio_value_stacked_by_ticker_pct_figure, kind="figure")))


			if _has_futures(types):
				specs.append(VisualSpec("stack_tickers_fut", "Futures — Stacked (Tickers)", _wrap_visual_builder(build_futures_value_stacked_by_ticker_figure, kind="figure")))
				specs.append(VisualSpec("stack_tickers_fut_pct", "% Futures — 100% Stacked (Tickers)", _wrap_visual_builder(build_futures_value_stacked_by_ticker_pct_figure, kind="figure")))

		if types is not None and ticker_volume_prices is not None and tickers is not None:
			specs.append(VisualSpec("stack_types", "Portfolio Value — Stacked (Types excl. futures + Cash + Credit)", _wrap_visual_builder(build_portfolio_value_stacked_by_type_figure, kind="figure")))
			specs.append(VisualSpec("stack_types_pct", "% Portfolio Value — 100% Stacked (Types excl. futures + Cash + Credit)", _wrap_visual_builder(build_portfolio_value_stacked_by_type_pct_figure, kind="figure")))
			if _has_futures(types):
				specs.append(VisualSpec("stack_types_fut", "Futures — Stacked (Types)", _wrap_visual_builder(build_futures_value_stacked_by_type_figure, kind="figure")))
				specs.append(VisualSpec("stack_types_fut_pct", "% Futures — 100% Stacked (Types)", _wrap_visual_builder(build_futures_value_stacked_by_type_pct_figure, kind="figure")))

	if dt_ok and tickers_profit is not None and tickers is not None:
		specs.append(VisualSpec("tickers_pnl", "PnL by Ticker (Lines)", _wrap_visual_builder(build_tickers_pnl_figure, kind="figure")))
		if ticker_volume_prices is not None:
			specs.append(VisualSpec("tickers_value", "Position Value by Ticker (Lines)", _wrap_visual_builder(build_tickers_value_figure, kind="figure")))
		specs.append(VisualSpec("tickers_table", "Ticker Summary Table", _wrap_visual_builder(build_ticker_summary_table_html, kind="html"), kind="html"))
		specs.append(VisualSpec("tickers_time", "Profit Table — Tickers × Time", _wrap_visual_builder(build_tickers_time_table_html, kind="html"), kind="html"))
		if ticker_volume_prices is not None:
			specs.append(VisualSpec("tickers_value_time", "Value Table — Tickers × Time", _wrap_visual_builder(build_tickers_value_time_table_html, kind="html"), kind="html"))

		specs.append(VisualSpec("pnl3d_tickers", "PnL — 3D Surface (Tickers)", _wrap_visual_builder(build_profit_3d_tickers_figure, kind="figure")))
		if ticker_volume_prices is not None:
			specs.append(VisualSpec("val3d_tickers", "Value — 3D Surface (Tickers)", _wrap_visual_builder(build_position_value_3d_tickers_figure, kind="figure")))

	if dt_ok and results_d[11] is not None and results_d[10] is not None:
		specs.append(VisualSpec("types", "Breakdown by Type (PnL)", _wrap_visual_builder(build_types_figure, kind="figure")))
		specs.append(VisualSpec("types_table", "Type Summary Table", _wrap_visual_builder(build_type_summary_table_html, kind="html"), kind="html"))
		specs.append(VisualSpec("types_time", "Profit Table — Types × Time", _wrap_visual_builder(build_types_time_table_html, kind="html"), kind="html"))
		if results_d[12] is not None:
			specs.append(VisualSpec("types_value_time", "Value Table — Types × Time", _wrap_visual_builder(build_types_value_time_table_html, kind="html"), kind="html"))

		specs.append(VisualSpec("pnl3d_types", "PnL — 3D Surface (Types)", _wrap_visual_builder(build_profit_3d_types_figure, kind="figure")))
		if results_d[12] is not None:
			specs.append(VisualSpec("val3d_types", "Value — 3D Surface (Types)", _wrap_visual_builder(build_position_value_3d_types_figure, kind="figure")))

	if dt_ok and tickers_profit is not None and tickers is not None:
		specs.append(VisualSpec("dist_tickers", "Final PnL Distribution — Tickers (Bars)", lambda r, b=None: build_distribution_tickers_bar_figure(_results_view(r, 0), b)))
		specs.append(VisualSpec("dist_tickers_hist", "Final PnL Distribution — Tickers (Histogram)", lambda r, b=None: build_distribution_tickers_hist_figure(_results_view(r, 0), b)))
	if dt_ok and results_d[11] is not None and results_d[10] is not None:
		specs.append(VisualSpec("dist_types", "Final PnL Distribution — Types (Bars)", lambda r, b=None: build_distribution_types_bar_figure(_results_view(r, 0), b)))
		specs.append(VisualSpec("dist_types_hist", "Final PnL Distribution — Types (Histogram)", lambda r, b=None: build_distribution_types_hist_figure(_results_view(r, 0), b)))



	return specs


def _is_future_type(t: Any) -> bool:
	try:
		s = str(t).strip().lower()
	except Exception:
		return False
	return s in ("future", "futures")


def _has_futures(types: Any) -> bool:
	if types is None:
		return False
	if isinstance(types, (list, tuple, np.ndarray)):
		return any(_is_future_type(t) for t in list(types))
	return _is_future_type(types)


def _add_3d_date_range_menu(fig: go.Figure, x_labels: List[str], *, x: float = 0.0, y: float = 1.22) -> None:
	if not x_labels:
		return
	idx = len(x_labels) - 1
	options = [
	 ("Dates: All", 0),
	 ("Dates: Last 30", max(0, idx - 29)),
	 ("Dates: Last 60", max(0, idx - 59)),
	 ("Dates: Last 90", max(0, idx - 89)),
	]
	buttons = []
	for label, start_i in options:
		buttons.append(
		 dict(
		  label=label,
		  method="relayout",
		  args=[{"scene.xaxis.range": [x_labels[start_i], x_labels[idx]]}],
		 )
		)
	fig.update_layout(
	 updatemenus=(list(fig.layout.updatemenus) if getattr(fig.layout, "updatemenus", None) else [])
	 + [dict(type="dropdown", x=x, y=y, xanchor="left", yanchor="top", buttons=buttons)],
	)


def _add_3d_subset_menu(
 fig: go.Figure,
 *,
 key: str,
 x_labels: List[str],
 all_labels: List[str],
 mat: np.ndarray,
 rank_values: np.ndarray,
 max_show: int = 80,
 x: float = 0.0,
 y: float = 1.12,
) -> None:
	if mat.size == 0 or not all_labels:
		return

	def _subset(n: int) -> Tuple[List[str], np.ndarray]:
		n = int(max(1, min(n, len(all_labels), max_show)))
		idx = _top_n_indices(rank_values, n=n)
		labels = [all_labels[i] for i in idx]
		return labels, mat[idx]

	choices = [(f"{key}: Top 10", 10), (f"{key}: Top 20", 20), (f"{key}: Top 50", 50), (f"{key}: All", len(all_labels))]
	buttons = []
	for label, n in choices:
		labels, m = _subset(n)
		buttons.append(
		 dict(
		  label=label,
		  method="update",
		  args=[
		   {"z": [m], "y": [labels], "x": [x_labels]},
		   {
		    "scene.yaxis.tickmode": "array",
		    "scene.yaxis.tickvals": labels,
		    "scene.yaxis.ticktext": [_shorten_label(t, max_len=24) for t in labels],
		   },
		  ],
		 )
		)

	fig.update_layout(
	 updatemenus=(list(fig.layout.updatemenus) if getattr(fig.layout, "updatemenus", None) else [])
	 + [dict(type="dropdown", x=x, y=y, xanchor="left", yanchor="top", buttons=buttons)],
	)


def build_ticker_summary_table_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	x = _to_datetime_index(r["lookup_days"])

	tickers = _as_str_list(r["tickers"])
	if tickers is None:
		return go.Figure(layout=dict(title=_safe_title(board, "Ticker Summary — no data")))

	tp = _as_2d_float(r["tickers_profit"], shape0=len(tickers), shape1=len(x))
	pv = _as_2d_float(r["ticker_volume_prices"], shape0=len(tickers), shape1=len(x))
	if tp is None:
		return go.Figure(layout=dict(title=_safe_title(board, "Ticker Summary — no data")))

	final_pnl = tp[:, -1]
	start_pnl = tp[:, 0]
	pnl_change = final_pnl - start_pnl
	max_pnl = np.nanmax(tp, axis=1)
	min_pnl = np.nanmin(tp, axis=1)

	if pv is not None:
		max_abs_pos = np.nanmax(np.abs(pv), axis=1)
		held_days = np.sum(np.abs(pv) > 0, axis=1)
	else:
		max_abs_pos = np.full(len(tickers), np.nan)
		held_days = np.full(len(tickers), np.nan)

	df = pd.DataFrame(
	 {
	  "ticker": tickers,
	  "final_pnl": final_pnl,
	  "pnl_change": pnl_change,
	  "max_pnl": max_pnl,
	  "min_pnl": min_pnl,
	  "max_abs_pos_value": max_abs_pos,
	  "held_days": held_days,
	 }
	)
	df = df.sort_values("final_pnl", ascending=False)

	fig = go.Figure(
	 data=[
	  go.Table(
	   header=dict(
	    values=[
	     "Ticker",
	     "Final PnL",
	     "PnL Δ",
	     "Max PnL",
	     "Min PnL",
	     "Max |PosValue|",
	     "Held Days",
	    ],
	    fill_color="#f0f0f0",
	    align="left",
	   ),
	   cells=dict(
	    values=[
	     df["ticker"],
	     np.round(df["final_pnl"], 2),
	     np.round(df["pnl_change"], 2),
	     np.round(df["max_pnl"], 2),
	     np.round(df["min_pnl"], 2),
	     np.round(df["max_abs_pos_value"], 2),
	     df["held_days"],
	    ],
	    align="left",
	   ),
	  )
	 ]
	)
	fig.update_layout(
	 title=_safe_title(board, "Trade Calculator — Ticker Summary"),
	 height=750,
	)
	return fig


def build_ticker_summary_table_html(results: List[Any], board: Optional[str] = None) -> str:
	r = _extract(results)
	x = _to_datetime_index(r["lookup_days"])

	tickers = _as_str_list(r["tickers"])
	if tickers is None:
		return _tabulator_html(title=_safe_title(board, "Ticker Summary — no data"), df=pd.DataFrame({"info": ["no data"]}))

	tp = _as_2d_float(r["tickers_profit"], shape0=len(tickers), shape1=len(x))
	pv = _as_2d_float(r["ticker_volume_prices"], shape0=len(tickers), shape1=len(x))
	if tp is None:
		return _tabulator_html(title=_safe_title(board, "Ticker Summary — no data"), df=pd.DataFrame({"info": ["no data"]}))

	final_pnl = tp[:, -1]
	start_pnl = tp[:, 0]
	pnl_change = final_pnl - start_pnl
	max_pnl = np.nanmax(tp, axis=1)
	min_pnl = np.nanmin(tp, axis=1)

	if pv is not None:
		max_abs_pos = np.nanmax(np.abs(pv), axis=1)
		held_days = np.sum(np.abs(pv) > 0, axis=1)
	else:
		max_abs_pos = np.full(len(tickers), np.nan)
		held_days = np.full(len(tickers), np.nan)

	df = pd.DataFrame(
	 {
	  "ticker": tickers,
	  "final_pnl": np.round(final_pnl, 2),
	  "pnl_change": np.round(pnl_change, 2),
	  "max_pnl": np.round(max_pnl, 2),
	  "min_pnl": np.round(min_pnl, 2),
	  "max_abs_pos_value": np.round(max_abs_pos, 2),
	  "held_days": held_days,
	 }
	)
	df = df.sort_values("final_pnl", ascending=False)


	try:
		cash_p = _as_1d_float(r.get("cash_profit_array", None), len(x))
		total_p = _as_1d_float(r.get("total_profit", None), len(x))
		summary_rows = []
		if total_p is not None:
			summary_rows.append(
			 {
			  "ticker": "Total Profit",
			  "final_pnl": round(float(total_p[-1]), 2),
			  "pnl_change": round(float(total_p[-1] - total_p[0]), 2),
			  "max_pnl": round(float(np.nanmax(total_p)), 2),
			  "min_pnl": round(float(np.nanmin(total_p)), 2),
			  "max_abs_pos_value": np.nan,
			  "held_days": np.nan,
			 }
			)
		if cash_p is not None:
			summary_rows.append(
			 {
			  "ticker": "Cash Profit",
			  "final_pnl": round(float(cash_p[-1]), 2),
			  "pnl_change": round(float(cash_p[-1] - cash_p[0]), 2),
			  "max_pnl": round(float(np.nanmax(cash_p)), 2),
			  "min_pnl": round(float(np.nanmin(cash_p)), 2),
			  "max_abs_pos_value": np.nan,
			  "held_days": np.nan,
			 }
			)
		if summary_rows:
			df = pd.concat([pd.DataFrame(summary_rows), df], ignore_index=True)
	except Exception:
		pass
	return _tabulator_html(title=_safe_title(board, "Trade Calculator — Ticker Summary"), df=df)


def build_type_summary_table_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	x = _to_datetime_index(r["lookup_days"])

	types_unique = _as_str_list(r["types_unique"])
	types_profits = r["types_profits"]
	if types_unique is None or types_profits is None:
		return go.Figure(layout=dict(title=_safe_title(board, "Type Summary — no data")))

	rows = []
	for name, arr in zip(types_unique, list(types_profits)):
		a = _as_1d_float(arr, len(x))
		if a is None:
			continue
		rows.append(
		 {
		  "type": str(name),
		  "final_pnl": float(a[-1]),
		  "max_pnl": float(np.nanmax(a)),
		  "min_pnl": float(np.nanmin(a)),
		 }
		)

	if not rows:
		return go.Figure(layout=dict(title=_safe_title(board, "Type Summary — no data")))

	df = pd.DataFrame(rows).sort_values("final_pnl", ascending=False)
	fig = go.Figure(
	 data=[
	  go.Table(
	   header=dict(values=["Type", "Final PnL", "Max PnL", "Min PnL"], fill_color="#f0f0f0", align="left"),
	   cells=dict(
	    values=[df["type"], np.round(df["final_pnl"], 2), np.round(df["max_pnl"], 2), np.round(df["min_pnl"], 2)],
	    align="left",
	   ),
	  )
	 ]
	)
	fig.update_layout(
	 title=_safe_title(board, "Trade Calculator — Type Summary"),
	 height=520,
	)
	return fig


def build_type_summary_table_html(results: List[Any], board: Optional[str] = None) -> str:
	r = _extract(results)
	x = _to_datetime_index(r["lookup_days"])

	types_unique = _as_str_list(r["types_unique"])
	types_profits = r["types_profits"]
	if types_unique is None or types_profits is None:
		return _tabulator_html(title=_safe_title(board, "Type Summary — no data"), df=pd.DataFrame({"info": ["no data"]}))

	rows = []
	for name, arr in zip(types_unique, list(types_profits)):
		a = _as_1d_float(arr, len(x))
		if a is None:
			continue
		rows.append(
		 {
		  "type": str(name),
		  "final_pnl": round(float(a[-1]), 2),
		  "max_pnl": round(float(np.nanmax(a)), 2),
		  "min_pnl": round(float(np.nanmin(a)), 2),
		 }
		)

	if not rows:
		return _tabulator_html(title=_safe_title(board, "Type Summary — no data"), df=pd.DataFrame({"info": ["no data"]}))

	df = pd.DataFrame(rows).sort_values("final_pnl", ascending=False)


	try:
		cash_p = _as_1d_float(r.get("cash_profit_array", None), len(x))
		total_p = _as_1d_float(r.get("total_profit", None), len(x))
		summary_rows2 = []
		if total_p is not None:
			summary_rows2.append(
			 {
			  "type": "Total Profit",
			  "final_pnl": round(float(total_p[-1]), 2),
			  "max_pnl": round(float(np.nanmax(total_p)), 2),
			  "min_pnl": round(float(np.nanmin(total_p)), 2),
			 }
			)
		if cash_p is not None:
			summary_rows2.append(
			 {
			  "type": "Cash Profit",
			  "final_pnl": round(float(cash_p[-1]), 2),
			  "max_pnl": round(float(np.nanmax(cash_p)), 2),
			  "min_pnl": round(float(np.nanmin(cash_p)), 2),
			 }
			)
		if summary_rows2:
			df = pd.concat([pd.DataFrame(summary_rows2), df], ignore_index=True)
	except Exception:
		pass
	return _tabulator_html(title=_safe_title(board, "Trade Calculator — Type Summary"), df=df)


def _extract(results: List[Any]) -> Dict[str, Any]:
	out: Dict[str, Any] = {}
	for i, name in enumerate(RESULT_FIELDS):
		if i < len(results):
			out[name] = results[i]
		else:
			out[name] = None
	return out


def build_overview_value_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	x = _to_datetime_index(r["lookup_days"])
	total_value = _as_1d_float(r["total_value"], len(x))
	fig = go.Figure()
	if total_value is not None:
		fig.add_trace(go.Scatter(x=x, y=total_value, name="Total Value", mode="lines"))
		try:
			m = float(np.nanmean(total_value)) if len(total_value) else 0.0
			md = float(np.nanmedian(total_value)) if len(total_value) else 0.0
			fig.add_trace(
			 go.Scatter(
			  x=x,
			  y=[m] * len(x),
			  name="Mean",
			  mode="lines",
			  line=dict(color="#888", dash="dash"),
			  visible="legendonly",
			  hovertemplate="Mean=%{y:.2f}<extra></extra>",
			 )
			)
			fig.add_trace(
			 go.Scatter(
			  x=x,
			  y=[md] * len(x),
			  name="Median",
			  mode="lines",
			  line=dict(color="#666", dash="dot"),
			  visible="legendonly",
			  hovertemplate="Median=%{y:.2f}<extra></extra>",
			 )
			)
		except Exception:
			pass
	fig.update_layout(
	 title=_safe_title(board, "Trade Calculator — Portfolio Value"),
	 hovermode="x unified",
	 height=650,
	 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
	 margin=dict(t=90, l=60, r=30, b=60),
	)
	fig.update_xaxes(rangeslider_visible=True)
	_add_theme_toggle(fig)
	return fig


def build_portfolio_value_all_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	x = _to_datetime_index(r["lookup_days"])
	xs = [d.strftime("%Y-%m-%d") for d in list(x)]
	step_ms = 24 * 60 * 60 * 1000
	try:
		xv = pd.to_datetime(pd.Series(list(x)), errors="coerce").dropna().sort_values().to_numpy(dtype="datetime64[ns]")
		if xv.size >= 2:
			diffs = np.diff(xv.astype("int64"))
			diffs = diffs[diffs > 0]
			if diffs.size:
				step_ms = int(float(np.median(diffs)) / 1_000_000.0)
				step_ms = int(max(3_600_000, min(step_ms, 90 * 24 * 60 * 60 * 1000)))
	except Exception:
		pass
	value = _as_1d_float(r.get("total_value", None), len(x))
	start_money = r.get("start_money", None)
	try:
		start_money_f = float(start_money)
	except Exception:
		start_money_f = float(value[0]) if value is not None and len(value) else 0.0

	if value is None or len(value) == 0:
		return go.Figure(layout=dict(title=_safe_title(board, "Portfolio Value — no data")))

	line = go.Scatter(x=xs, y=value, name="Portfolio Value", mode="lines")


	open_ = np.concatenate([[start_money_f], value[:-1]])
	close_ = value
	high_ = np.maximum(open_, close_)
	low_ = np.minimum(open_, close_)
	candle = go.Candlestick(x=xs, open=open_, high=high_, low=low_, close=close_, name="Candle")
	try:
		# Make candles span the active period (especially noticeable for weekly/monthly).
		candle.update(xperiod=step_ms, xperiodalignment="middle")
	except Exception:
		pass


	prev = np.concatenate([[start_money_f], value[:-1]])
	deltas = (value - prev).astype(float)
	waterfall = go.Waterfall(
	 x=list(xs),
	 y=list(deltas),
	 measure=["relative"] * len(deltas),
	 base=float(start_money_f),
	 name="Δ Value",
	 connector={"line": {"color": "#888"}},
	 increasing={"marker": {"color": "#2ca02c"}},
	 decreasing={"marker": {"color": "#d62728"}},
	)
	try:
		waterfall.update(width=float(step_ms) * 0.65)
	except Exception:
		pass
	start_marker = go.Scatter(
	 x=[xs[0]],
	 y=[float(start_money_f)],
	 mode="markers+text",
	 name="Start",
	 text=["Start"],
	 textposition="top center",
	 marker={"size": 8, "color": "#1f77b4"},
	 showlegend=False,
	)
	end_marker = go.Scatter(
	 x=[xs[-1]],
	 y=[float(value[-1])],
	 mode="markers+text",
	 name="End",
	 text=["End"],
	 textposition="top center",
	 marker={"size": 8, "color": "#1f77b4"},
	 showlegend=False,
	)

	fig = go.Figure(data=[line, candle, waterfall, start_marker, end_marker])

	try:
		m = float(np.nanmean(value)) if len(value) else 0.0
		md = float(np.nanmedian(value)) if len(value) else 0.0
		mean_line = go.Scatter(x=xs, y=[m] * len(xs), name="Mean", mode="lines", line=dict(color="#888", dash="dash"), visible="legendonly")
		med_line = go.Scatter(x=xs, y=[md] * len(xs), name="Median", mode="lines", line=dict(color="#666", dash="dot"), visible="legendonly")
		fig.add_trace(mean_line)
		fig.add_trace(med_line)
	except Exception:
		pass

	fig.data[0].visible = True
	fig.data[1].visible = False
	fig.data[2].visible = False
	fig.data[3].visible = False
	fig.data[4].visible = False
	if len(fig.data) > 5:
		fig.data[5].visible = "legendonly"
	if len(fig.data) > 6:
		fig.data[6].visible = "legendonly"


	try:
		vis_line = [True, False, False, False, False] + (["legendonly", "legendonly"] if len(fig.data) >= 7 else [])
		vis_candle = [False, True, False, False, False] + ([False, False] if len(fig.data) >= 7 else [])
		vis_wf = [False, False, True, True, True] + ([False, False] if len(fig.data) >= 7 else [])
		view_meta = {
		 "label": "View:",
		 "default_index": 0,
		 "views": [
		  {"label": "Line", "visible": vis_line, "relayout": {"yaxis.title.text": "Value", "xaxis.rangeslider.visible": True, "xaxis.type": "date"}},
		  {"label": "Candle", "visible": vis_candle, "relayout": {"yaxis.title.text": "Value", "xaxis.rangeslider.visible": True, "xaxis.type": "date"}},
		  {"label": "Waterfall", "visible": vis_wf, "relayout": {"yaxis.title.text": "Δ Value", "xaxis.rangeslider.visible": False, "xaxis.type": "date", "bargap": 0.55, "bargroupgap": 0.0}},
		 ],
		}
	except Exception:
		view_meta = None

	fig.update_layout(
	 title=_safe_title(board, "Trade Calculator — Portfolio Value"),
	 height=900,
	 hovermode="x unified",
	 margin=dict(t=120, l=60, r=30, b=70),
	 meta=({"tc_view_switch": view_meta} if view_meta else None),
	)
	fig.update_xaxes(rangeslider_visible=True)
	_add_theme_toggle(fig)
	return fig


def build_overview_flows_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	x = _to_datetime_index(r["lookup_days"])
	total_value = _as_1d_float(r.get("total_value", None), len(x))
	total_profit = _as_1d_float(r["total_profit"], len(x))
	total_cash = _as_1d_float(r["total_cash"], len(x))
	total_credit = _as_1d_float(r["total_credit"], len(x))
	cash_profit_array = _as_1d_float(r["cash_profit_array"], len(x))
	fig = go.Figure()
	if total_value is not None:
		fig.add_trace(go.Scatter(x=x, y=total_value, name="Portfolio Value", mode="lines"))
	if total_profit is not None:
		fig.add_trace(go.Scatter(x=x, y=total_profit, name="Total Profit", mode="lines"))
	if cash_profit_array is not None:
		fig.add_trace(go.Scatter(x=x, y=cash_profit_array, name="Cash Flows", mode="lines"))
	if total_cash is not None:
		fig.add_trace(go.Scatter(x=x, y=total_cash, name="Cash", mode="lines"))
	if total_credit is not None:
		fig.add_trace(go.Scatter(x=x, y=total_credit, name="Credit", mode="lines"))
	fig.update_layout(
	 title=_safe_title(board, "Trade Calculator — Profit / Cash / Credit"),
	 hovermode="x unified",
	 height=700,
	 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
	 margin=dict(t=100, l=60, r=30, b=60),
	)
	fig.update_xaxes(rangeslider_visible=True)
	_add_theme_toggle(fig)
	return fig


def _stack_components_by_ticker(
 results: List[Any],
 *,
 mode: str,
 include_cash_credit: bool,
) -> Tuple[pd.DatetimeIndex, List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	r = _extract(results)
	x = _to_datetime_index(r["lookup_days"])

	tickers = _as_str_list(r["tickers"]) or []
	types = _as_str_list(r["types"], length=len(tickers)) or ["unknown" for _ in tickers]
	pos = _as_2d_float(r["ticker_volume_prices"], shape0=len(tickers), shape1=len(x))
	if pos is None:
		pos = np.zeros((len(tickers), len(x)), dtype=float)

	mask = np.array([_is_future_type(t) for t in types], dtype=bool)
	if mode == "non_futures":
		keep = ~mask
	elif mode == "futures":
		keep = mask
	else:
		keep = np.ones(len(tickers), dtype=bool)

	tickers = [t for t, k in zip(tickers, keep) if k]
	pos = pos[keep] if pos.shape[0] else pos

	if include_cash_credit:
		cash = _as_1d_float(r["total_cash"], len(x))
		credit = _as_1d_float(r["total_credit"], len(x))
		if cash is None:
			cash = np.zeros(len(x), dtype=float)
		if credit is None:
			credit = np.zeros(len(x), dtype=float)
		credit_neg = -credit
	else:
		cash = np.zeros(len(x), dtype=float)
		credit_neg = np.zeros(len(x), dtype=float)

	total_from_components = (cash + credit_neg + np.nansum(pos, axis=0)).astype(float)
	return x, tickers, pos, cash, credit_neg, total_from_components


def _stack_components_by_type(
 results: List[Any],
 *,
 mode: str,
 include_cash_credit: bool,
) -> Tuple[pd.DatetimeIndex, List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	r = _extract(results)
	x = _to_datetime_index(r["lookup_days"])

	tickers = _as_str_list(r["tickers"]) or []
	types = _as_str_list(r["types"], length=len(tickers)) or ["unknown" for _ in tickers]
	pos = _as_2d_float(r["ticker_volume_prices"], shape0=len(tickers), shape1=len(x))
	if pos is None:
		pos = np.zeros((len(tickers), len(x)), dtype=float)

	mask_fut = np.array([_is_future_type(t) for t in types], dtype=bool)
	if mode == "non_futures":
		keep = ~mask_fut
	elif mode == "futures":
		keep = mask_fut
	else:
		keep = np.ones(len(tickers), dtype=bool)

	types_f = [t for t, k in zip(types, keep) if k]
	pos_f = pos[keep] if pos.shape[0] else pos


	uniq = sorted(set(types_f))
	by_type = np.zeros((len(uniq), len(x)), dtype=float)
	idx_map: Dict[str, int] = {t: i for i, t in enumerate(uniq)}
	for i, t in enumerate(types_f):
		by_type[idx_map[t]] += pos_f[i]

	if include_cash_credit:
		cash = _as_1d_float(r["total_cash"], len(x))
		credit = _as_1d_float(r["total_credit"], len(x))
		if cash is None:
			cash = np.zeros(len(x), dtype=float)
		if credit is None:
			credit = np.zeros(len(x), dtype=float)
		credit_neg = -credit
	else:
		cash = np.zeros(len(x), dtype=float)
		credit_neg = np.zeros(len(x), dtype=float)

	total_from_components = (cash + credit_neg + np.nansum(by_type, axis=0)).astype(float)
	return x, uniq, by_type, cash, credit_neg, total_from_components


def _build_portfolio_stacked(
 *,
 x: pd.DatetimeIndex,
 labels: List[str],
 values: np.ndarray,
 cash: np.ndarray,
 credit_neg: np.ndarray,
 total_value: np.ndarray,
 board: Optional[str],
 title: str,
 percent: bool,
) -> go.Figure:
	if values.size:
		order = np.argsort(np.abs(values[:, -1]))[::-1]
		values = values[order]
		labels = [labels[i] for i in order]


	xs = [d.strftime("%Y-%m-%d") for d in list(x)]


	values_use = values
	cash_use = cash
	credit_use = credit_neg
	if percent:




		pos_sum = np.maximum(cash_use.astype(float), 0.0) + np.nansum(np.maximum(values_use.astype(float), 0.0), axis=0)
		neg_sec_abs = -np.nansum(np.minimum(values_use.astype(float), 0.0), axis=0)
		cred_abs = np.abs(np.asarray(credit_use, dtype=float))
		excess_credit = np.maximum(0.0, cred_abs - neg_sec_abs)
		den2 = (pos_sum - excess_credit).copy()
		zero_mask = np.abs(den2) < 1e-12
		den2[zero_mask] = 1.0
		values_show = (values_use / den2) * 100.0
		cash_show = (cash_use / den2) * 100.0

		credit_show = (credit_use / den2) * 100.0
		if values_show.size:
			values_show[:, zero_mask] = 0.0
		cash_show[zero_mask] = 0.0
		credit_show[zero_mask] = 0.0
		overlay_y = np.full(len(xs), 100.0)
		y_title = "%"
	else:
		values_show = values_use
		cash_show = cash_use
		credit_show = credit_use
		overlay_y = total_value.astype(float) if total_value is not None else (cash_use + credit_use + np.nansum(values_use, axis=0)).astype(float)
		y_title = "Value"


	area_traces: List[go.BaseTraceType] = []
	bar_traces: List[go.BaseTraceType] = []

	def _pos(a: np.ndarray) -> np.ndarray:
		return np.maximum(np.asarray(a, dtype=float), 0.0)

	def _neg(a: np.ndarray) -> np.ndarray:
		return np.minimum(np.asarray(a, dtype=float), 0.0)



	LEG_TOTAL = 0
	LEG_CASH = 10
	LEG_CREDIT = 20
	LEG_COMP0 = 30


	credit_overlay_y = np.asarray(credit_show, dtype=float)
	cash_neg_y = _neg(cash_show)
	cash_pos_y = _pos(cash_show)


	area_traces.append(
	 go.Scatter(
	  x=xs,
	  y=cash_neg_y,
	  name="Cash",
	  legendgroup="Cash",
	  legendrank=LEG_CASH,
	  showlegend=False,
	  mode="lines",
	  stackgroup="tc_neg",
	  hovertemplate=("%{x}<br>Cash=%{y:.2f}%<extra></extra>" if percent else "%{x}<br>Cash=%{y:.2f}<extra></extra>"),
	 )
	)
	for i, (name, series) in enumerate(zip(labels, list(values_show))):
		area_traces.append(
		 go.Scatter(
		  x=xs,
		  y=_neg(series),
		  name=name,
		  legendgroup=name,
		  legendrank=LEG_COMP0 + i,
		  showlegend=False,
		  mode="lines",
		  stackgroup="tc_neg",
		  hovertemplate=("%{x}<br>" + name + "=%{y:.2f}%<extra></extra>" if percent else "%{x}<br>" + name + "=%{y:.2f}<extra></extra>"),
		 )
		)


	area_traces.append(
	 go.Scatter(
	  x=xs,
	  y=credit_overlay_y,
	  name="Credit",
	  legendgroup="Credit",
	  legendrank=LEG_CREDIT,
	  showlegend=True,
	  mode="lines",
	  fill="tozeroy",
	  hovertemplate=("%{x}<br>Credit=%{y:.2f}%<extra></extra>" if percent else "%{x}<br>Credit=%{y:.2f}<extra></extra>"),
	 )
	)


	area_traces.append(
	 go.Scatter(
	  x=xs,
	  y=cash_pos_y,
	  name="Cash",
	  legendgroup="Cash",
	  legendrank=LEG_CASH,
	  showlegend=True,
	  mode="lines",
	  stackgroup="tc_pos",
	  hovertemplate=("%{x}<br>Cash=%{y:.2f}%<extra></extra>" if percent else "%{x}<br>Cash=%{y:.2f}<extra></extra>"),
	 )
	)
	for i, (name, series) in enumerate(zip(labels, list(values_show))):
		area_traces.append(
		 go.Scatter(
		  x=xs,
		  y=_pos(series),
		  name=name,
		  legendgroup=name,
		  legendrank=LEG_COMP0 + i,
		  showlegend=True,
		  mode="lines",
		  stackgroup="tc_pos",
		  hovertemplate=("%{x}<br>" + name + "=%{y:.2f}%<extra></extra>" if percent else "%{x}<br>" + name + "=%{y:.2f}<extra></extra>"),
		 )
		)


	bar_traces.append(
	 go.Bar(
	  x=xs,
	  y=cash_show,
	  name="Cash",
	  legendgroup="Cash",
	  legendrank=LEG_CASH,
	  hovertemplate=("%{x}<br>Cash=%{y:.2f}%<extra></extra>" if percent else "%{x}<br>Cash=%{y:.2f}<extra></extra>"),
	 )
	)
	for i, (name, series) in enumerate(zip(labels, list(values_show))):
		bar_traces.append(
		 go.Bar(
		  x=xs,
		  y=series,
		  name=name,
		  legendgroup=name,
		  legendrank=LEG_COMP0 + i,
		  hovertemplate=("%{x}<br>" + name + "=%{y:.2f}%<extra></extra>" if percent else "%{x}<br>" + name + "=%{y:.2f}<extra></extra>"),
		 )
		)
	bar_traces.append(
	 go.Bar(
	  x=xs,
	  y=credit_show,
	  name="Credit",
	  legendgroup="Credit",
	  legendrank=LEG_CREDIT,
	  base=0,
	  opacity=0.6,
	  hovertemplate=("%{x}<br>Credit=%{y:.2f}%<extra></extra>" if percent else "%{x}<br>Credit=%{y:.2f}<extra></extra>"),
	 )
	)


	overlay = go.Scatter(
	 x=xs,
	 y=overlay_y,
	 name="Total" if not percent else "Total (100%)",
	 legendgroup="Total",
	 legendrank=LEG_TOTAL,
	 mode="lines",
	 line=dict(color="#111111", width=2),
	 hovertemplate=("%{x}<br>Total=%{y:.2f}%<extra></extra>" if percent else "%{x}<br>Total=%{y:.2f}<extra></extra>"),
	)


	fig = go.Figure(data=area_traces + [overlay] + bar_traces)
	area_count = len(area_traces)
	bar_count = len(bar_traces)

	vis_area = [True] * area_count + [True] + [False] * bar_count
	vis_bar = [False] * area_count + [True] + [True] * bar_count
	fig.update_traces(visible=False)
	for i, v in enumerate(vis_area):
		fig.data[i].visible = v

	fig.update_layout(
	 title=_safe_title(board, title),
	 hovermode="x unified",
	 height=850,
	 barmode="relative",
	 bargap=0.4,
	 bargroupgap=0.0,
	 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
	 margin=dict(t=120, l=60, r=30, b=70),
	)

	try:
		comp_count = len(labels)


		area_cash_neg_idx = 0
		area_comp_neg_start = 1
		area_credit_overlay_idx = area_comp_neg_start + comp_count
		area_cash_pos_idx = area_credit_overlay_idx + 1
		area_comp_pos_start = area_cash_pos_idx + 1
		overlay_idx = area_comp_pos_start + comp_count

		bar_cash_idx = overlay_idx + 1
		bar_comp_start = bar_cash_idx + 1
		bar_credit_idx = bar_comp_start + comp_count
		fig.update_layout(
		 meta={
		  "tc_stack_full": {
		   "x": list(xs),
		   "labels": list(labels),
		   "values": values.astype(float).tolist() if hasattr(values, "tolist") else [],
		   "cash": cash.astype(float).tolist() if hasattr(cash, "tolist") else [],
		   "credit_neg": credit_neg.astype(float).tolist() if hasattr(credit_neg, "tolist") else [],
		   "total": total_value.astype(float).tolist() if hasattr(total_value, "tolist") else [],
		   "percent": bool(percent),
		   "views": {
		    "area": {
		     "visible": list(vis_area),
		     "relayout": {"xaxis.type": "date", "xaxis.rangeslider.visible": True, "bargap": 0.4, "bargroupgap": 0.0},
		    },
		    "bar": {
		     "visible": list(vis_bar),
		     "relayout": {"xaxis.type": "category", "xaxis.rangeslider.visible": False, "bargap": 0.4, "bargroupgap": 0.0},
		    },
		   },
		   "trace_map": {
		    "area_cash_neg": area_cash_neg_idx,
		    "area_cash_pos": area_cash_pos_idx,
		    "area_comp_neg_start": area_comp_neg_start,
		    "area_comp_pos_start": area_comp_pos_start,
		    "area_credit": area_credit_overlay_idx,
		    "overlay": overlay_idx,
		    "bar_credit": bar_credit_idx,
		    "bar_cash": bar_cash_idx,
		    "bar_comp_start": bar_comp_start,
		    "comp_count": comp_count,
		   },
		  }
		 }
		)
	except Exception:
		pass
	fig.update_xaxes(rangeslider_visible=True)
	fig.update_yaxes(title_text=y_title)
	_add_theme_toggle(fig, x=1.0, y=1.12)
	return fig


def build_portfolio_value_stacked_by_ticker_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	x, labels, values, cash, credit_neg, _ = _stack_components_by_ticker(results, mode="non_futures", include_cash_credit=True)
	total_value = _as_1d_float(r.get("total_value", None), len(x))
	return _build_portfolio_stacked(
	 x=x,
	 labels=labels,
	 values=values,
	 cash=cash,
	 credit_neg=credit_neg,
	 total_value=total_value,
	 board=board,
	 title="Portfolio Value — Components by Ticker (+Cash, -Credit)",
	 percent=False,
	)


def build_portfolio_value_stacked_by_ticker_pct_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	x, labels, values, cash, credit_neg, _ = _stack_components_by_ticker(results, mode="non_futures", include_cash_credit=True)
	total_value = _as_1d_float(r.get("total_value", None), len(x))
	return _build_portfolio_stacked(
	 x=x,
	 labels=labels,
	 values=values,
	 cash=cash,
	 credit_neg=credit_neg,
	 total_value=total_value,
	 board=board,
	 title="% Portfolio Value — Components by Ticker (+Cash, -Credit)",
	 percent=True,
	)


def build_portfolio_value_stacked_by_type_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	x, labels, values, cash, credit_neg, _ = _stack_components_by_type(results, mode="non_futures", include_cash_credit=True)
	total_value = _as_1d_float(r.get("total_value", None), len(x))
	return _build_portfolio_stacked(
	 x=x,
	 labels=labels,
	 values=values,
	 cash=cash,
	 credit_neg=credit_neg,
	 total_value=total_value,
	 board=board,
	 title="Portfolio Value — Components by Type (+Cash, -Credit)",
	 percent=False,
	)


def build_portfolio_value_stacked_by_type_pct_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	x, labels, values, cash, credit_neg, _ = _stack_components_by_type(results, mode="non_futures", include_cash_credit=True)
	total_value = _as_1d_float(r.get("total_value", None), len(x))
	return _build_portfolio_stacked(
	 x=x,
	 labels=labels,
	 values=values,
	 cash=cash,
	 credit_neg=credit_neg,
	 total_value=total_value,
	 board=board,
	 title="% Portfolio Value — Components by Type (+Cash, -Credit)",
	 percent=True,
	)


def build_futures_value_stacked_by_ticker_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	x, labels, values, cash, credit_neg, total_value = _stack_components_by_ticker(results, mode="futures", include_cash_credit=False)
	return _build_portfolio_stacked(
	 x=x,
	 labels=labels,
	 values=values,
	 cash=cash,
	 credit_neg=credit_neg,
	 total_value=total_value,
	 board=board,
	 title="Futures Value — Components by Ticker",
	 percent=False,
	)


def build_futures_value_stacked_by_ticker_pct_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	x, labels, values, cash, credit_neg, total_value = _stack_components_by_ticker(results, mode="futures", include_cash_credit=False)
	return _build_portfolio_stacked(
	 x=x,
	 labels=labels,
	 values=values,
	 cash=cash,
	 credit_neg=credit_neg,
	 total_value=total_value,
	 board=board,
	 title="% Futures Value — Components by Ticker",
	 percent=True,
	)


def build_futures_value_stacked_by_type_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	x, labels, values, cash, credit_neg, total_value = _stack_components_by_type(results, mode="futures", include_cash_credit=False)
	return _build_portfolio_stacked(
	 x=x,
	 labels=labels,
	 values=values,
	 cash=cash,
	 credit_neg=credit_neg,
	 total_value=total_value,
	 board=board,
	 title="Futures Value — Components by Type",
	 percent=False,
	)


def build_futures_value_stacked_by_type_pct_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	x, labels, values, cash, credit_neg, total_value = _stack_components_by_type(results, mode="futures", include_cash_credit=False)
	return _build_portfolio_stacked(
	 x=x,
	 labels=labels,
	 values=values,
	 cash=cash,
	 credit_neg=credit_neg,
	 total_value=total_value,
	 board=board,
	 title="% Futures Value — Components by Type",
	 percent=True,
	)



def _tabulator_html(*, title: str, df: pd.DataFrame) -> str:
	data = df.to_dict(orient="records")
	columns = []
	for i, c in enumerate(df.columns):
		col = {"title": str(c), "field": str(c), "headerFilter": True}
		if i == 0:
			col["frozen"] = True
			col["hozAlign"] = "left"
		columns.append(col)
	data_json = json.dumps(data, ensure_ascii=False)
	cols_json = json.dumps(columns, ensure_ascii=False)

	return f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>{title}</title>
  <link href='https://unpkg.com/tabulator-tables@5.6.1/dist/css/tabulator.min.css' rel='stylesheet'>
  <style>
		:root {{
			--bg:#ffffff; --fg:#111; --muted:#666; --border:rgba(0,0,0,0.14);
			--panel:#f7f7f7; --panel2:#f1f1f1; --hover:#ececec; --active:#dfe7ff;
			--accent:#2f6feb;
		}}
		body.dark {{
			--bg:#0f0f10; --fg:#f0f0f0; --muted:#aaa; --border:rgba(255,255,255,0.16);
			--panel:#171718; --panel2:#1e1e1f; --hover:#242425; --active:#1f2a4a;
			--accent:#7aa2ff;
		}}
		html, body {{ height: 100%; }}
		body {{ margin: 0; padding: 10px 12px; font-family: Segoe UI, Arial, sans-serif; background:var(--bg); color:var(--fg); }}
		.header {{ padding: 10px 12px; border: 1px solid var(--border); border-radius: 12px; background: var(--panel); display:flex; gap:12px; align-items:center; }}
		.title {{ display:none; }}
		.note {{ color:var(--muted); font-size:12px; flex: 1; }}
		.wrap {{ padding: 10px 0 0 0; height: calc(100vh - 86px); overflow: hidden; }}
		#grid {{ height: 100%; }}
		.tabulator .tabulator-cell {{ user-select: text; }}
		.tabulator .tabulator-tableholder {{ overflow-x: auto !important; }}

		.tabulator {{ border: 1px solid var(--border); border-radius: 12px; overflow: hidden; background: var(--bg); min-width: 100%; }}
		.tabulator .tabulator-tableholder {{ overflow-x: auto; }}
		.tabulator .tabulator-header {{ background: var(--panel2); border-bottom: 1px solid var(--border); }}
		.tabulator .tabulator-header .tabulator-col {{ background: var(--panel2); color: var(--fg); border-right: 1px solid var(--border); }}
		.tabulator .tabulator-header .tabulator-col.tabulator-sortable:hover {{ background: var(--hover); }}
		.tabulator .tabulator-header .tabulator-col .tabulator-col-title {{ font-weight: 600; }}
		.tabulator .tabulator-header .tabulator-col input {{
			width: 100%; padding: 6px 8px; margin-top: 6px;
			border-radius: 10px; border: 1px solid var(--border);
			background: var(--bg); color: var(--fg);
			outline: none;
		}}
		.tabulator .tabulator-row {{ background: var(--bg); color: var(--fg); border-bottom: 1px solid var(--border); }}
		.tabulator .tabulator-row.tabulator-row-even {{ background: color-mix(in srgb, var(--bg) 94%, var(--panel) 6%); }}
		.tabulator .tabulator-row:hover {{ background: var(--hover); }}
		.tabulator .tabulator-cell {{ border-right: 1px solid var(--border); }}
		.tabulator .tabulator-footer {{ background: var(--panel2); border-top: 1px solid var(--border); color: var(--fg); }}
  </style>
</head>
<body>
	<div class='header'>
		<div class='title'></div>
		<div class='note'>Tip: drag to select, Ctrl+C to copy. Use header filters to search</div>
	</div>
	<div class='wrap'>
		<div id='grid'></div>
	</div>

  <script src='https://unpkg.com/tabulator-tables@5.6.1/dist/js/tabulator.min.js'></script>
  <script>
    const tableData = {data_json};
    const columns = {cols_json};
    let grid;

		function _downloadBlob(filename, mime, text) {{
			try {{
				const blob = new Blob([text], {{ type: mime || 'application/octet-stream' }});
				const url = URL.createObjectURL(blob);
				const a = document.createElement('a');
				a.href = url;
				a.download = filename || 'export';
				a.style.display = 'none';
				document.body.appendChild(a);
				a.click();
				setTimeout(() => {{
					try {{ URL.revokeObjectURL(url); }} catch (e) {{}}
					try {{ a.remove(); }} catch (e) {{}}
				}}, 120);
				return true;
			}} catch (e) {{
				return false;
			}}
		}}

		function _loadScript(src) {{
			return new Promise((resolve, reject) => {{
				try {{
					const s = document.createElement('script');
					s.src = src;
					s.async = true;
					s.onload = () => resolve(true);
					s.onerror = () => reject(new Error('Failed to load: ' + src));
					document.head.appendChild(s);
				}} catch (e) {{
					reject(e);
				}}
			}});
		}}

		async function _ensureXlsx() {{
			if (window.XLSX) return true;
			try {{
				await _loadScript('https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js');
				return !!window.XLSX;
			}} catch (e) {{
				return false;
			}}
		}}

		async function _ensurePdf() {{
			if (window.jsPDF) return true;
			try {{
				await _loadScript('https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js');
				try {{
					if (!window.jsPDF && window.jspdf && window.jspdf.jsPDF) window.jsPDF = window.jspdf.jsPDF;
				}} catch (e) {{}}
				await _loadScript('https://cdnjs.cloudflare.com/ajax/libs/jspdf-autotable/3.5.31/jspdf.plugin.autotable.min.js');
				return !!window.jsPDF;
			}} catch (e) {{
				return false;
			}}
		}}

		function _exportCapsTabulator() {{
			const ok = !!(grid && window.Tabulator);
			return {{
				kind: 'tabulator',
				supported: ok,
				data: ok ? ['csv', 'json', 'xlsx', 'pdf', 'html'] : [],
				clipboard: ok ? ['csv', 'json'] : [],
				other: ok ? ['html'] : [],
			}};
		}}

		async function _copyToClipboard(text) {{
			try {{
				if (navigator && navigator.clipboard && navigator.clipboard.writeText) {{
					await navigator.clipboard.writeText(text);
					return true;
				}}
			}} catch (e) {{}}
			return false;
		}}

		async function _handleExportMessage(msg) {{
			try {{
				if (!msg || !msg.action) return;
				if (msg.action === 'caps') {{
					try {{ window.parent && window.parent.postMessage({{ tc_export_caps: _exportCapsTabulator() }}, '*'); }} catch (e) {{}}
					return;
				}}
				if (msg.action !== 'run') return;
				const kind = String(msg.kind || '');
				if (kind && kind !== 'tabulator') return;
				if (!grid) return;
				const scope = String(msg.scope || '');
				const fmt = String(msg.format || '');
				const baseName = (msg.filename == null) ? 'export' : String(msg.filename);

				if (scope === 'data') {{
					if (fmt === 'csv') {{
						try {{ grid.download('csv', baseName + '.csv'); }} catch (e) {{}}
						return;
					}}
					if (fmt === 'json') {{
						try {{ grid.download('json', baseName + '.json'); }} catch (e) {{}}
						return;
					}}
					if (fmt === 'html') {{
						try {{ grid.download('html', baseName + '.html'); }} catch (e) {{}}
						return;
					}}
					if (fmt === 'xlsx') {{
						const ok = await _ensureXlsx();
						if (!ok) return;
						try {{ grid.download('xlsx', baseName + '.xlsx', {{ sheetName: 'Data' }}); }} catch (e) {{}}
						return;
					}}
					if (fmt === 'pdf') {{
						const ok = await _ensurePdf();
						if (!ok) return;
						try {{ grid.download('pdf', baseName + '.pdf', {{ orientation: 'landscape', title: baseName }}); }} catch (e) {{}}
						return;
					}}
				}}

				if (scope === 'clipboard') {{
					if (fmt === 'json') {{
						try {{ await _copyToClipboard(JSON.stringify(grid.getData() || [], null, 2)); }} catch (e) {{}}
						return;
					}}
					if (fmt === 'csv') {{
						try {{
							const cols = (grid.getColumns ? grid.getColumns() : []).map(c => (c && c.getField) ? c.getField() : null).filter(Boolean);
							const rows = grid.getData() || [];
							function esc(v) {{
								const s = (v == null) ? '' : String(v);
								if (/[\\n\\r,\\"]/g.test(s)) return '"' + s.replace(/\\"/g, '""') + '"';
								return s;
							}}
							const lines = [];
							lines.push(cols.map(esc).join(','));
							rows.forEach(r => {{ lines.push(cols.map(c => esc(r[c])).join(',')); }});
							await _copyToClipboard(lines.join('\\n'));
						}} catch (e) {{}}
						return;
					}}
				}}

				if (scope === 'other' && fmt === 'html') {{
					try {{
						const html = document.documentElement.outerHTML;
						_downloadBlob(baseName + '.html', 'text/html;charset=utf-8', html);
					}} catch (e) {{}}
					return;
				}}
			}} catch (e) {{}}
		}}

    function init() {{
      grid = new Tabulator('#grid', {{
        data: tableData,
        columns: columns,
				height: '100%',
				layout: 'fitDataFill',
				responsiveLayout: false,
				columnMinWidth: 110,
        reactiveData: false,
				selectable: false,
				selectableRange: 1,
				selectableRangeColumns: true,
				selectableRangeRows: true,
				selectableRangeClearCells: true,
        clipboard: true,
        clipboardCopyStyled: false,
        clipboardCopyConfig: {{ rowHeaders: false, columnHeaders: true }},
        movableColumns: true,
        resizableColumns: true,
        headerSort: true,
				columnDefaults: {{ headerFilter: true, headerFilterLiveFilter: true }},
      }});
    }}

		function applyTheme(dark) {{
			document.body.classList.toggle('dark', !!dark);
		}}

		window.addEventListener('message', (ev) => {{
			if (!ev || !ev.data) return;
			if (typeof ev.data.tc_theme === 'boolean') applyTheme(ev.data.tc_theme);
			if (ev.data && ev.data.tc_export) _handleExportMessage(ev.data.tc_export);
		}});

    init();
		try {{
			const savedTheme = localStorage.getItem('tc_theme');
			applyTheme(savedTheme === 'dark');
		}} catch (e) {{ applyTheme(false); }}
		try {{
			window.parent && window.parent.postMessage({{ tc_export_caps: _exportCapsTabulator() }}, '*');
		}} catch (e) {{}}
  </script>
</body>
</html>"""


def _matrix_to_df(*, row_name: str, row_labels: List[str], col_labels: List[str], mat: np.ndarray) -> pd.DataFrame:
	cols = [row_name] + list(col_labels)
	recs = []
	for i, rlab in enumerate(row_labels):
		row = {row_name: rlab}
		for j, clab in enumerate(col_labels):
			val = float(mat[i, j]) if mat.size else float("nan")
			row[clab] = round(val, 6)
		recs.append(row)
	return pd.DataFrame.from_records(recs, columns=cols)


def _prepend_summary_rows(
 df: pd.DataFrame,
 *,
 row_name: str,
 col_labels: List[str],
 rows: List[Tuple[str, np.ndarray]],
) -> pd.DataFrame:
	if df is None or df.empty:
		base_cols = [row_name] + list(col_labels)
		df = pd.DataFrame(columns=base_cols)
	recs: List[Dict[str, Any]] = []
	for label, arr in rows:
		if arr is None:
			continue
		a = _as_1d_float(arr, len(col_labels))
		if a is None:
			continue
		row: Dict[str, Any] = {row_name: str(label)}
		for j, clab in enumerate(col_labels):
			row[clab] = round(float(a[j]), 6)
		recs.append(row)
	if not recs:
		return df
	add = pd.DataFrame.from_records(recs, columns=[row_name] + list(col_labels))
	return pd.concat([add, df], ignore_index=True)


def build_tickers_time_table_html(results: List[Any], board: Optional[str] = None) -> str:
	r = _extract(results)
	dates = _to_datetime_index(r["lookup_days"])
	date_cols = [d.strftime("%Y-%m-%d") if not pd.isna(d) else "NaT" for d in dates]
	labels = _as_str_list(r["tickers"]) or []
	mat = _as_2d_float(r["tickers_profit"], shape0=len(labels), shape1=len(dates))
	if mat is None:
		return _tabulator_html(title=_safe_title(board, "Profit — Tickers × Time (no data)"), df=pd.DataFrame({"info": ["no data"]}))
	df = _matrix_to_df(row_name="Ticker", row_labels=labels, col_labels=date_cols, mat=np.round(mat, 6))
	total_profit = _as_1d_float(r.get("total_profit", None), len(dates))
	cash_profit = _as_1d_float(r.get("cash_profit_array", None), len(dates))
	rows = []
	if total_profit is not None:
		rows.append(("Total Profit", total_profit))
	if cash_profit is not None:
		rows.append(("Cash Profit", cash_profit))
	if rows:
		df = _prepend_summary_rows(df, row_name="Ticker", col_labels=date_cols, rows=rows)
	return _tabulator_html(title=_safe_title(board, "Profit — Tickers × Time"), df=df)


def build_types_time_table_html(results: List[Any], board: Optional[str] = None) -> str:
	r = _extract(results)
	dates = _to_datetime_index(r["lookup_days"])
	date_cols = [d.strftime("%Y-%m-%d") if not pd.isna(d) else "NaT" for d in dates]
	labels = _as_str_list(r["types_unique"]) or []
	tp = r.get("types_profits", None)
	if tp is None:
		return _tabulator_html(title=_safe_title(board, "Profit — Types × Time (no data)"), df=pd.DataFrame({"info": ["no data"]}))
	rows = []
	for arr in list(tp):
		a = _as_1d_float(arr, len(dates))
		rows.append(a if a is not None else np.full(len(dates), np.nan))
	mat = np.array(rows, dtype=float)
	df = _matrix_to_df(row_name="Type", row_labels=labels, col_labels=date_cols, mat=np.round(mat, 6))
	total_profit = _as_1d_float(r.get("total_profit", None), len(dates))
	cash_profit = _as_1d_float(r.get("cash_profit_array", None), len(dates))
	rows2 = []
	if total_profit is not None:
		rows2.append(("Total Profit", total_profit))
	if cash_profit is not None:
		rows2.append(("Cash Profit", cash_profit))
	if rows2:
		df = _prepend_summary_rows(df, row_name="Type", col_labels=date_cols, rows=rows2)
	return _tabulator_html(title=_safe_title(board, "Profit — Types × Time"), df=df)


def build_tickers_value_time_table_html(results: List[Any], board: Optional[str] = None) -> str:
	r = _extract(results)
	dates = _to_datetime_index(r["lookup_days"])
	date_cols = [d.strftime("%Y-%m-%d") if not pd.isna(d) else "NaT" for d in dates]
	labels = _as_str_list(r["tickers"]) or []
	mat = _as_2d_float(r["ticker_volume_prices"], shape0=len(labels), shape1=len(dates))
	if mat is None:
		return _tabulator_html(title=_safe_title(board, "Value — Tickers × Time (no data)"), df=pd.DataFrame({"info": ["no data"]}))
	df = _matrix_to_df(row_name="Ticker", row_labels=labels, col_labels=date_cols, mat=np.round(mat, 6))
	total_value = _as_1d_float(r.get("total_value", None), len(dates))
	total_cash = _as_1d_float(r.get("total_cash", None), len(dates))
	total_credit = _as_1d_float(r.get("total_credit", None), len(dates))
	df = _prepend_summary_rows(
	 df,
	 row_name="Ticker",
	 col_labels=date_cols,
	 rows=[
	  ("Total", total_value) if total_value is not None else ("Total", None),
	  ("Cash", total_cash) if total_cash is not None else ("Cash", None),
	  ("Credit", total_credit) if total_credit is not None else ("Credit", None),
	 ],
	)
	return _tabulator_html(title=_safe_title(board, "Value — Tickers × Time"), df=df)


def build_types_value_time_table_html(results: List[Any], board: Optional[str] = None) -> str:
	r = _extract(results)
	dates = _to_datetime_index(r["lookup_days"])
	date_cols = [d.strftime("%Y-%m-%d") if not pd.isna(d) else "NaT" for d in dates]
	labels = _as_str_list(r["types_unique"]) or []
	tp = r.get("types_volume_prices", None)
	if tp is None:
		return _tabulator_html(title=_safe_title(board, "Value — Types × Time (no data)"), df=pd.DataFrame({"info": ["no data"]}))
	rows = []
	for arr in list(tp):
		a = _as_1d_float(arr, len(dates))
		rows.append(a if a is not None else np.full(len(dates), np.nan))
	mat = np.array(rows, dtype=float)
	df = _matrix_to_df(row_name="Type", row_labels=labels, col_labels=date_cols, mat=np.round(mat, 6))
	total_value = _as_1d_float(r.get("total_value", None), len(dates))
	total_cash = _as_1d_float(r.get("total_cash", None), len(dates))
	total_credit = _as_1d_float(r.get("total_credit", None), len(dates))
	df = _prepend_summary_rows(
	 df,
	 row_name="Type",
	 col_labels=date_cols,
	 rows=[
	  ("Total", total_value) if total_value is not None else ("Total", None),
	  ("Cash", total_cash) if total_cash is not None else ("Cash", None),
	  ("Credit", total_credit) if total_credit is not None else ("Credit", None),
	 ],
	)
	return _tabulator_html(title=_safe_title(board, "Value — Types × Time"), df=df)


def build_portfolio_value_waterfall_candle_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	x = _to_datetime_index(r["lookup_days"])
	step_ms = 24 * 60 * 60 * 1000
	try:
		xv = pd.to_datetime(pd.Series(list(x)), errors="coerce").dropna().sort_values().to_numpy(dtype="datetime64[ns]")
		if xv.size >= 2:
			diffs = np.diff(xv.astype("int64"))
			diffs = diffs[diffs > 0]
			if diffs.size:
				step_ms = int(float(np.median(diffs)) / 1_000_000.0)
				step_ms = int(max(3_600_000, min(step_ms, 90 * 24 * 60 * 60 * 1000)))
	except Exception:
		pass
	value = _as_1d_float(r["total_value"], len(x))
	start_money = r.get("start_money", None)
	try:
		start_money_f = float(start_money)
	except Exception:
		start_money_f = float(value[0]) if value is not None and len(value) else 0.0

	if value is None or len(value) == 0:
		return go.Figure(layout=dict(title=_safe_title(board, "Portfolio Value — no data")))


 # IMPORTANT: lengths must match; otherwise Plotly may render an extra final drop.
	prev = np.concatenate([[start_money_f], value[:-1]])
	deltas = (value - prev).astype(float)
	x_wf = ["Start"] + [d.strftime("%Y-%m-%d") for d in x] + ["End"]
	measure = ["absolute"] + ["relative"] * len(deltas) + ["total"]


	y_wf = np.concatenate([[start_money_f], deltas, [float(value[-1])]])

	waterfall = go.Waterfall(
	 x=x_wf,
	 y=y_wf,
	 measure=measure,
	 name="Waterfall",
	 connector={"line": {"color": "#888"}},
	)


	open_ = np.concatenate([[start_money_f], value[:-1]])
	close_ = value
	high_ = np.maximum(open_, close_)
	low_ = np.minimum(open_, close_)
	candle = go.Candlestick(
	 x=x,
	 open=open_,
	 high=high_,
	 low=low_,
	 close=close_,
	 name="Candle",
	)
	try:
		candle.update(xperiod=step_ms, xperiodalignment="middle")
	except Exception:
		pass


	fig = go.Figure(data=[candle, waterfall])
	fig.data[0].visible = True
	fig.data[1].visible = False

	fig.update_layout(
	 title=_safe_title(board, "Portfolio Value — Waterfall / Candle"),
	 height=750,
	 hovermode="x",
	 margin=dict(t=110, l=60, r=30, b=60),
	 meta={
	  "tc_view_switch": {
	   "label": "View:",
	   "default_index": 0,
	   "views": [
	    {"label": "Candle", "visible": [True, False]},
	    {"label": "Waterfall", "visible": [False, True]},
	   ],
	  }
	 },
	)
	_add_theme_toggle(fig)
	return fig


def build_portfolio_value_pct_delta_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	x = _to_datetime_index(r["lookup_days"])
	xs = [d.strftime("%Y-%m-%d") for d in list(x)]
	value = _as_1d_float(r["total_value"], len(x))
	if value is None or len(value) == 0:
		return go.Figure(layout=dict(title=_safe_title(board, "% Delta — no data")))

	prev = np.roll(value, 1)
	prev[0] = np.nan
	prev[np.isclose(prev, 0.0)] = np.nan
	pct = (value / prev - 1.0) * 100.0

	fig = go.Figure()
	fig.add_trace(
	 go.Scatter(
	  x=xs,
	  y=pct,
	  mode="lines+markers",
	  name="% Δ from previous",
	  hovertemplate="%{x|%Y-%m-%d}<br>%Δ=%{y:.3f}%<extra></extra>",
	 )
	)

	try:
		m = float(np.nanmean(pct))
		md = float(np.nanmedian(pct))
		fig.add_trace(
		 go.Scatter(
		  x=xs,
		  y=[m] * len(xs),
		  name="Mean",
		  mode="lines",
		  line=dict(color="#888", dash="dash"),
		  visible="legendonly",
		  hovertemplate="Mean=%{y:.3f}%<extra></extra>",
		 )
		)
		fig.add_trace(
		 go.Scatter(
		  x=xs,
		  y=[md] * len(xs),
		  name="Median",
		  mode="lines",
		  line=dict(color="#666", dash="dot"),
		  visible="legendonly",
		  hovertemplate="Median=%{y:.3f}%<extra></extra>",
		 )
		)
	except Exception:
		pass
	fig.add_hline(y=0.0, line_color="#888", line_width=1)
	fig.update_layout(
	 title=_safe_title(board, "% Delta of Portfolio Value"),
	 height=520,
	 hovermode="x unified",
	 margin=dict(t=90, l=60, r=30, b=60),
	)
	fig.update_xaxes(rangeslider_visible=True)
	fig.update_yaxes(title_text="%")
	_add_theme_toggle(fig)
	return fig


def build_tickers_pnl_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	x = _to_datetime_index(r["lookup_days"])
	cash_profit_array = _as_1d_float(r.get("cash_profit_array", None), len(x))

	tickers = _as_str_list(r["tickers"])
	if tickers is None:
		return go.Figure(layout=dict(title=_safe_title(board, "Tickers — no data")))

	tp = _as_2d_float(r["tickers_profit"], shape0=len(tickers), shape1=len(x))
	pv = _as_2d_float(r["ticker_volume_prices"], shape0=len(tickers), shape1=len(x))

	fig = go.Figure()


	if cash_profit_array is not None:
		fig.add_trace(
		 go.Scatter(
		  x=x,
		  y=cash_profit_array,
		  name="Cash Profit",
		  mode="lines",
		  line=dict(width=2),
		  hovertemplate="%{x|%Y-%m-%d}<br>CashProfit=%{y:.2f}<extra></extra>",
		 )
		)
	if tp is not None:
		try:
			mean_y = np.nanmean(tp, axis=0)
			med_y = np.nanmedian(tp, axis=0)
			fig.add_trace(
			 go.Scatter(
			  x=x,
			  y=mean_y,
			  name="Mean (tickers)",
			  mode="lines",
			  line=dict(color="#888", dash="dash"),
			  visible="legendonly",
			  hovertemplate="%{x|%Y-%m-%d}<br>MeanPnL=%{y:.2f}<extra></extra>",
			 )
			)
			fig.add_trace(
			 go.Scatter(
			  x=x,
			  y=med_y,
			  name="Median (tickers)",
			  mode="lines",
			  line=dict(color="#666", dash="dot"),
			  visible="legendonly",
			  hovertemplate="%{x|%Y-%m-%d}<br>MedianPnL=%{y:.2f}<extra></extra>",
			 )
			)
		except Exception:
			pass


	if tp is not None:
		final = tp[:, -1]
		show_idx = set(_top_n_indices(final, n=12).tolist())
		for i, t in enumerate(tickers):
			fig.add_trace(
			 go.Scatter(
			  x=x,
			  y=tp[i],
			  name=f"{t}",
			  mode="lines",
			  visible=True if i in show_idx else "legendonly",
			  hovertemplate="%{x|%Y-%m-%d}<br>PnL=%{y:.2f}<extra>" + t + "</extra>",
			 ),
			)


	topn_meta = None
	if tp is not None and len(tickers) > 0:
		final = tp[:, -1]
		n_t = len(tickers)
		prefix = len(fig.data) - n_t
		try:
			topn_meta = {
			 "label": "Show:",
			 "trace_start": int(prefix),
			 "trace_count": int(n_t),
			 "rank": [float(abs(v)) for v in list(final)],
			 "options": [12, 25, 0],
			 "default_n": 12,
			}
		except Exception:
			topn_meta = None

	fig.update_layout(
	 title=_safe_title(board, "Trade Calculator — PnL by Ticker"),
	 hovermode="x unified",
	 height=800,
	 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
	 margin=dict(t=120, l=60, r=30, b=70),
	 meta=({"tc_topn": topn_meta} if topn_meta else None),
	)
	fig.update_xaxes(rangeslider_visible=True)
	_add_theme_toggle(fig)
	return fig


def build_tickers_value_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	x = _to_datetime_index(r["lookup_days"])

	tickers = _as_str_list(r["tickers"])
	if tickers is None:
		return go.Figure(layout=dict(title=_safe_title(board, "Position Value by Ticker — no data")))

	pv = _as_2d_float(r["ticker_volume_prices"], shape0=len(tickers), shape1=len(x))
	if pv is None:
		return go.Figure(layout=dict(title=_safe_title(board, "Position Value by Ticker — no data")))

	fig = go.Figure()

	try:
		mean_y = np.nanmean(pv, axis=0)
		med_y = np.nanmedian(pv, axis=0)
		fig.add_trace(
		 go.Scatter(
		  x=x,
		  y=mean_y,
		  name="Mean (tickers)",
		  mode="lines",
		  line=dict(color="#888", dash="dash"),
		  visible="legendonly",
		  hovertemplate="%{x|%Y-%m-%d}<br>MeanValue=%{y:.2f}<extra></extra>",
		 )
		)
		fig.add_trace(
		 go.Scatter(
		  x=x,
		  y=med_y,
		  name="Median (tickers)",
		  mode="lines",
		  line=dict(color="#666", dash="dot"),
		  visible="legendonly",
		  hovertemplate="%{x|%Y-%m-%d}<br>MedianValue=%{y:.2f}<extra></extra>",
		 )
		)
	except Exception:
		pass
	max_abs = np.nanmax(np.abs(pv), axis=1)
	show_idx = set(_top_n_indices(max_abs, n=12).tolist())
	for i, t in enumerate(tickers):
		fig.add_trace(
		 go.Scatter(
		  x=x,
		  y=pv[i],
		  name=f"{t}",
		  mode="lines",
		  visible=True if i in show_idx else "legendonly",
		  opacity=0.8,
		  hovertemplate="%{x|%Y-%m-%d}<br>Value=%{y:.2f}<extra>" + t + "</extra>",
		 )
		)

	topn_meta = None
	if len(tickers) > 0:
		n_t = len(tickers)
		prefix = len(fig.data) - n_t
		try:
			topn_meta = {
			 "label": "Show:",
			 "trace_start": int(prefix),
			 "trace_count": int(n_t),
			 "rank": [float(abs(v)) for v in list(max_abs)],
			 "options": [12, 25, 0],
			 "default_n": 12,
			}
		except Exception:
			topn_meta = None

	fig.update_layout(
	 title=_safe_title(board, "Trade Calculator — Position Value by Ticker"),
	 hovermode="x unified",
	 height=800,
	 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
	 margin=dict(t=120, l=60, r=30, b=70),
	 meta=({"tc_topn": topn_meta} if topn_meta else None),
	)
	fig.update_xaxes(rangeslider_visible=True)
	_add_theme_toggle(fig)
	return fig


def build_types_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	x = _to_datetime_index(r["lookup_days"])
	cash_profit_array = _as_1d_float(r.get("cash_profit_array", None), len(x))

	types_unique = _as_str_list(r["types_unique"])
	types_profits = r["types_profits"]
	if types_unique is None or types_profits is None:
		return go.Figure(layout=dict(title=_safe_title(board, "Types — no data")))


	series: List[np.ndarray] = []
	valid_names: List[str] = []
	for name, arr in zip(types_unique, list(types_profits)):
		a = _as_1d_float(arr, len(x))
		if a is None:
			continue
		valid_names.append(str(name))
		series.append(a)

	fig = go.Figure()

	if cash_profit_array is not None:
		fig.add_trace(
		 go.Scatter(
		  x=x,
		  y=cash_profit_array,
		  name="Cash Profit",
		  mode="lines",
		  line=dict(width=2),
		  hovertemplate="%{x|%Y-%m-%d}<br>CashProfit=%{y:.2f}<extra></extra>",
		 )
		)


	if series:
		try:
			mat = np.vstack([np.asarray(a, dtype=float) for a in series])
			mean_y = np.nanmean(mat, axis=0)
			med_y = np.nanmedian(mat, axis=0)
			fig.add_trace(
			 go.Scatter(
			  x=x,
			  y=mean_y,
			  mode="lines",
			  name="Mean (types)",
			  line=dict(color="#888", dash="dash"),
			  visible="legendonly",
			  hovertemplate="%{x|%Y-%m-%d}<br>MeanPnL=%{y:.2f}<extra></extra>",
			 )
			)
			fig.add_trace(
			 go.Scatter(
			  x=x,
			  y=med_y,
			  mode="lines",
			  name="Median (types)",
			  line=dict(color="#666", dash="dot"),
			  visible="legendonly",
			  hovertemplate="%{x|%Y-%m-%d}<br>MedianPnL=%{y:.2f}<extra></extra>",
			 )
			)
		except Exception:
			pass
	for name, a in zip(valid_names, series):
		fig.add_trace(
		 go.Scatter(
		  x=x,
		  y=a,
		  mode="lines",
		  name=name,
		  hovertemplate="%{x|%Y-%m-%d}<br>PnL=%{y:.2f}<extra>" + name + "</extra>",
		 )
		)

	fig.update_layout(
	 title=_safe_title(board, "Trade Calculator — Types"),
	 hovermode="x unified",
	 height=650,
	 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
	 margin=dict(t=100, l=60, r=30, b=70),
	)
	fig.update_xaxes(rangeslider_visible=True)
	_add_theme_toggle(fig)
	return fig


def build_distribution_tickers_bar_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	tickers = _as_str_list(r["tickers"])
	if tickers is None:
		return go.Figure(layout=dict(title=_safe_title(board, "Final PnL — Tickers (no data)")))

	tp = np.array(r["tickers_profit"], dtype=float) if r["tickers_profit"] is not None else None
	if tp is None or tp.ndim != 2 or tp.shape[0] != len(tickers):
		return go.Figure(layout=dict(title=_safe_title(board, "Distributions — no data")))


	x = _to_datetime_index(r["lookup_days"])
	x_labels = [d.strftime("%Y-%m-%d") if not pd.isna(d) else "NaT" for d in x]
	cash_profit_array = _as_1d_float(r.get("cash_profit_array", None), len(x))

	rows = [(tickers[i], float(tp[i, -1])) for i in range(tp.shape[0])]
	if cash_profit_array is not None and len(cash_profit_array) == len(x):
		rows.append(("Cash Profit", float(cash_profit_array[-1])))
	rows.sort(key=lambda t: t[1])
	tickers_sorted = [t[0] for t in rows]
	final_sorted = np.array([t[1] for t in rows], dtype=float)
	colors = ["#1f77b4" if n == "Cash Profit" else ("#2ca02c" if v >= 0 else "#d62728") for n, v in rows]
	fig = go.Figure(
	 data=[
	  go.Bar(
	   x=tickers_sorted,
	   y=final_sorted,
	   marker_color=colors,
	   name="Final PnL",
	   hovertemplate="%{x}<br>FinalPnL=%{y:.2f}<extra></extra>",
	  )
	 ]
	)

	try:
		m = float(np.nanmean(final_sorted)) if final_sorted.size else 0.0
		md = float(np.nanmedian(final_sorted)) if final_sorted.size else 0.0
		fig.add_trace(
		 go.Scatter(
		  x=tickers_sorted,
		  y=[m] * len(tickers_sorted),
		  mode="lines",
		  name="Mean",
		  line=dict(color="#888", dash="dash"),
		  visible="legendonly",
		  hovertemplate="Mean=%{y:.2f}<extra></extra>",
		 )
		)
		fig.add_trace(
		 go.Scatter(
		  x=tickers_sorted,
		  y=[md] * len(tickers_sorted),
		  mode="lines",
		  name="Median",
		  line=dict(color="#666", dash="dot"),
		  visible="legendonly",
		  hovertemplate="Median=%{y:.2f}<extra></extra>",
		 )
		)
	except Exception:
		pass
	fig.update_layout(
	 title=_safe_title(board, "Trade Calculator — Final PnL by Ticker"),
	 height=620,
	 hovermode="closest",
	 margin=dict(t=80, l=60, r=30, b=80),
	)
	fig.update_xaxes(tickangle=45)
	try:
		if cash_profit_array is not None and len(cash_profit_array) == len(x):
			tp2 = np.vstack([tp, cash_profit_array.reshape(1, -1)])
			labs = list(tickers) + ["Cash Profit"]
			fig.update_layout(meta={"tc_dist_full": {"labels": labs, "dates": x_labels, "mat": tp2.tolist()}})
		else:
			fig.update_layout(meta={"tc_dist_full": {"labels": list(tickers), "dates": x_labels, "mat": tp.tolist()}})
	except Exception:
		pass
	_add_theme_toggle(fig)
	return fig


def build_distribution_tickers_hist_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	tickers = _as_str_list(r["tickers"])
	if tickers is None:
		return go.Figure(layout=dict(title=_safe_title(board, "Final PnL Histogram — Tickers (no data)")))
	tp = np.array(r["tickers_profit"], dtype=float) if r["tickers_profit"] is not None else None
	if tp is None or tp.ndim != 2 or tp.shape[0] != len(tickers):
		return go.Figure(layout=dict(title=_safe_title(board, "Final PnL Histogram — Tickers (no data)")))
	final = tp[:, -1]

	try:
		x = _to_datetime_index(r["lookup_days"])
		cash_profit_array = _as_1d_float(r.get("cash_profit_array", None), len(x))
		if cash_profit_array is not None and len(cash_profit_array) == len(x):
			final = np.concatenate([final.astype(float), np.array([float(cash_profit_array[-1])], dtype=float)])
	except Exception:
		pass

	try:
		counts, edges = np.histogram(final[np.isfinite(final)], bins=40)
	except Exception:
		counts, edges = np.histogram(np.array([0.0]), bins=40)
	centers = (edges[:-1] + edges[1:]) / 2.0
	maxc = float(np.max(counts)) if counts.size else 1.0
	fig = go.Figure(
	 data=[
	  go.Bar(
	   x=centers,
	   y=counts,
	   name="Count",
	   hovertemplate="PnL=%{x:.2f}<br>Count=%{y}<extra></extra>",
	  )
	 ]
	)
	try:
		m = float(np.nanmean(final)) if final.size else 0.0
		md = float(np.nanmedian(final)) if final.size else 0.0
		fig.add_trace(
		 go.Scatter(
		  x=[m, m],
		  y=[0, maxc],
		  mode="lines",
		  name="Mean",
		  line=dict(color="#888", dash="dash"),
		  visible="legendonly",
		  hovertemplate="Mean=%{x:.2f}<extra></extra>",
		 )
		)
		fig.add_trace(
		 go.Scatter(
		  x=[md, md],
		  y=[0, maxc],
		  mode="lines",
		  name="Median",
		  line=dict(color="#666", dash="dot"),
		  visible="legendonly",
		  hovertemplate="Median=%{x:.2f}<extra></extra>",
		 )
		)
	except Exception:
		pass
	fig.update_layout(
	 title=_safe_title(board, "Trade Calculator — Final PnL Histogram (Tickers)"),
	 height=600,
	 hovermode="closest",
	 margin=dict(t=80, l=60, r=30, b=60),
	)
	fig.update_xaxes(title_text="Final PnL")
	fig.update_yaxes(title_text="Count")
	_add_theme_toggle(fig)
	return fig


def build_distribution_types_bar_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	types_unique = _as_str_list(r["types_unique"])
	tp = r.get("types_profits", None)
	if types_unique is None or tp is None:
		return go.Figure(layout=dict(title=_safe_title(board, "Final PnL — Types (no data)")))
	x = _to_datetime_index(r["lookup_days"])
	cash_profit_array = _as_1d_float(r.get("cash_profit_array", None), len(x))
	rows = []
	mat_rows: List[np.ndarray] = []
	for name, arr in zip(types_unique, list(tp)):
		a = _as_1d_float(arr, len(x))
		if a is None:
			continue
		rows.append((str(name), float(a[-1])))
		mat_rows.append(a)
	if not rows:
		return go.Figure(layout=dict(title=_safe_title(board, "Final PnL — Types (no data)")))
	if cash_profit_array is not None and len(cash_profit_array) == len(x):
		rows.append(("Cash Profit", float(cash_profit_array[-1])))
		mat_rows.append(cash_profit_array.astype(float))

	mat = np.array(mat_rows, dtype=float) if mat_rows else np.zeros((0, len(x)), dtype=float)
	x_labels = [d.strftime("%Y-%m-%d") if not pd.isna(d) else "NaT" for d in x]
	rows.sort(key=lambda t: t[1])
	labels = [r[0] for r in rows]
	vals = np.array([r[1] for r in rows], dtype=float)
	colors = ["#1f77b4" if n == "Cash Profit" else ("#2ca02c" if v >= 0 else "#d62728") for n, v in rows]
	fig = go.Figure(
	 data=[go.Bar(x=labels, y=vals, marker_color=colors, name="Final PnL", hovertemplate="%{x}<br>FinalPnL=%{y:.2f}<extra></extra>")]
	)

	try:
		m = float(np.nanmean(vals)) if vals.size else 0.0
		md = float(np.nanmedian(vals)) if vals.size else 0.0
		fig.add_trace(
		 go.Scatter(
		  x=labels,
		  y=[m] * len(labels),
		  mode="lines",
		  name="Mean",
		  line=dict(color="#888", dash="dash"),
		  visible="legendonly",
		  hovertemplate="Mean=%{y:.2f}<extra></extra>",
		 )
		)
		fig.add_trace(
		 go.Scatter(
		  x=labels,
		  y=[md] * len(labels),
		  mode="lines",
		  name="Median",
		  line=dict(color="#666", dash="dot"),
		  visible="legendonly",
		  hovertemplate="Median=%{y:.2f}<extra></extra>",
		 )
		)
	except Exception:
		pass
	fig.update_layout(
	 title=_safe_title(board, "Trade Calculator — Final PnL by Type"),
	 height=600,
	 margin=dict(t=80, l=60, r=30, b=70),
	)
	fig.update_xaxes(tickangle=30)
	try:
		labs = list(types_unique)
		if cash_profit_array is not None and len(cash_profit_array) == len(x):
			labs = labs + ["Cash Profit"]
		fig.update_layout(meta={"tc_dist_full": {"labels": labs, "dates": x_labels, "mat": mat.tolist()}})
	except Exception:
		pass
	_add_theme_toggle(fig)
	return fig


def build_distribution_types_hist_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	types_unique = _as_str_list(r["types_unique"])
	tp = r.get("types_profits", None)
	if types_unique is None or tp is None:
		return go.Figure(layout=dict(title=_safe_title(board, "Final PnL Histogram — Types (no data)")))
	x = _to_datetime_index(r["lookup_days"])
	finals = []
	for arr in list(tp):
		a = _as_1d_float(arr, len(x))
		if a is None:
			continue
		finals.append(float(a[-1]))

	try:
		cash_profit_array = _as_1d_float(r.get("cash_profit_array", None), len(x))
		if cash_profit_array is not None and len(cash_profit_array) == len(x):
			finals.append(float(cash_profit_array[-1]))
	except Exception:
		pass
	if not finals:
		return go.Figure(layout=dict(title=_safe_title(board, "Final PnL Histogram — Types (no data)")))
	arr = np.array(finals, dtype=float)
	try:
		counts, edges = np.histogram(arr[np.isfinite(arr)], bins=30)
	except Exception:
		counts, edges = np.histogram(np.array([0.0]), bins=30)
	centers = (edges[:-1] + edges[1:]) / 2.0
	maxc = float(np.max(counts)) if counts.size else 1.0
	fig = go.Figure(
	 data=[
	  go.Bar(
	   x=centers,
	   y=counts,
	   name="Count",
	   hovertemplate="PnL=%{x:.2f}<br>Count=%{y}<extra></extra>",
	  )
	 ]
	)
	try:
		m = float(np.nanmean(arr)) if arr.size else 0.0
		md = float(np.nanmedian(arr)) if arr.size else 0.0
		fig.add_trace(
		 go.Scatter(
		  x=[m, m],
		  y=[0, maxc],
		  mode="lines",
		  name="Mean",
		  line=dict(color="#888", dash="dash"),
		  visible="legendonly",
		  hovertemplate="Mean=%{x:.2f}<extra></extra>",
		 )
		)
		fig.add_trace(
		 go.Scatter(
		  x=[md, md],
		  y=[0, maxc],
		  mode="lines",
		  name="Median",
		  line=dict(color="#666", dash="dot"),
		  visible="legendonly",
		  hovertemplate="Median=%{x:.2f}<extra></extra>",
		 )
		)
	except Exception:
		pass
	fig.update_layout(
	 title=_safe_title(board, "Trade Calculator — Final PnL Histogram (Types)"),
	 height=600,
	 margin=dict(t=80, l=60, r=30, b=60),
	)
	fig.update_xaxes(title_text="Final PnL")
	fig.update_yaxes(title_text="Count")
	_add_theme_toggle(fig)
	return fig



def build_position_value_3d_tickers_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	x = _to_datetime_index(r["lookup_days"])

	tickers = _as_str_list(r["tickers"])
	if tickers is None:
		return go.Figure(layout=dict(title=_safe_title(board, "Position Heatmap — no data")))

	pv = _as_2d_float(r["ticker_volume_prices"], shape0=len(tickers), shape1=len(x))
	if pv is None:
		return go.Figure(layout=dict(title=_safe_title(board, "Position Value — 3D Surface (Tickers) — no data")))


	x_labels = [d.strftime("%Y-%m-%d") if not pd.isna(d) else "NaT" for d in x]


	max_abs = np.nanmax(np.abs(pv), axis=1)
	show_idx = _top_n_indices(max_abs, n=min(20, len(tickers)))
	pv2 = pv[show_idx]
	tickers2 = [tickers[i] for i in show_idx]

	fig = go.Figure(
	 data=[
	  go.Surface(
	   z=pv2,
	   x=x_labels,
	   y=tickers2,
	   colorscale="YlGnBu",
	   colorbar=dict(title="Value"),
	   hovertemplate="Ticker=%{y}<br>Date=%{x}<br>Value=%{z:.2f}<extra></extra>",
	  )
	 ]
	)

	try:
		mean_z = np.nanmean(pv2, axis=0)
		med_z = np.nanmedian(pv2, axis=0)
		z_mean = np.tile(mean_z, (len(tickers2), 1))
		z_med = np.tile(med_z, (len(tickers2), 1))
		fig.add_trace(
		 go.Surface(
		  z=z_mean,
		  x=x_labels,
		  y=tickers2,
		  name="Mean",
		  showlegend=True,
		  showscale=False,
		  opacity=0.32,
		  colorscale="Greys",
		  visible="legendonly",
		  hovertemplate="Mean=%{z:.2f}<extra></extra>",
		 )
		)
		fig.add_trace(
		 go.Surface(
		  z=z_med,
		  x=x_labels,
		  y=tickers2,
		  name="Median",
		  showlegend=True,
		  showscale=False,
		  opacity=0.32,
		  colorscale="Greys",
		  visible="legendonly",
		  hovertemplate="Median=%{z:.2f}<extra></extra>",
		 )
		)
	except Exception:
		pass

	try:
		fig.update_layout(meta={"tc_surface_full": {"x": x_labels, "y": list(tickers), "z": pv.tolist()}})
	except Exception:
		pass

	fig.update_layout(
	 title=_safe_title(board, "Trade Calculator — Position Value 3D (Tickers, top by |value|)"),
	 height=900,
	 scene=dict(
	  xaxis=dict(
	   title="Date",
	   tickmode="array",
	   tickvals=x_labels[:: max(1, len(x_labels) // 12)],
	   ticktext=x_labels[:: max(1, len(x_labels) // 12)],
	  ),
	  yaxis=dict(
	   title="Ticker",
	   tickmode="array",
	   tickvals=tickers2,
	   ticktext=[_shorten_label(t, max_len=24) for t in tickers2],
	  ),
	  zaxis=dict(title="Value"),
	 ),
	)
	_add_theme_toggle(fig)
	return fig


def build_position_value_3d_types_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	x = _to_datetime_index(r["lookup_days"])

	types_unique = _as_str_list(r["types_unique"])
	tp = r.get("types_volume_prices", None)
	if types_unique is None or tp is None:
		return go.Figure(layout=dict(title=_safe_title(board, "Position Value — 3D Surface (Types) — no data")))

	rows: List[np.ndarray] = []
	labels: List[str] = []
	for name, arr in zip(types_unique, list(tp)):
		a = _as_1d_float(arr, len(x))
		if a is None:
			continue
		labels.append(str(name))
		rows.append(a)

	if not rows:
		return go.Figure(layout=dict(title=_safe_title(board, "Position Value — 3D Surface (Types) — no data")))

	mat = np.array(rows, dtype=float)
	x_labels = [d.strftime("%Y-%m-%d") if not pd.isna(d) else "NaT" for d in x]
	labels2 = labels

	fig = go.Figure(
	 data=[
	  go.Surface(
	   z=mat,
	   x=x_labels,
	   y=labels2,
	   colorscale="YlGnBu",
	   colorbar=dict(title="Value"),
	   hovertemplate="Type=%{y}<br>Date=%{x}<br>Value=%{z:.2f}<extra></extra>",
	  )
	 ]
	)

	try:
		mean_z = np.nanmean(mat, axis=0)
		med_z = np.nanmedian(mat, axis=0)
		z_mean = np.tile(mean_z, (len(labels2), 1))
		z_med = np.tile(med_z, (len(labels2), 1))
		fig.add_trace(
		 go.Surface(
		  z=z_mean,
		  x=x_labels,
		  y=labels2,
		  name="Mean",
		  showlegend=True,
		  showscale=False,
		  opacity=0.32,
		  colorscale="Greys",
		  visible="legendonly",
		  hovertemplate="Mean=%{z:.2f}<extra></extra>",
		 )
		)
		fig.add_trace(
		 go.Surface(
		  z=z_med,
		  x=x_labels,
		  y=labels2,
		  name="Median",
		  showlegend=True,
		  showscale=False,
		  opacity=0.32,
		  colorscale="Greys",
		  visible="legendonly",
		  hovertemplate="Median=%{z:.2f}<extra></extra>",
		 )
		)
	except Exception:
		pass

	try:
		fig.update_layout(meta={"tc_surface_full": {"x": x_labels, "y": labels2, "z": mat.tolist()}})
	except Exception:
		pass

	fig.update_layout(
	 title=_safe_title(board, "Trade Calculator — Position Value 3D (Types)"),
	 height=850,
	 scene=dict(
	  xaxis=dict(
	   title="Date",
	   tickmode="array",
	   tickvals=x_labels[:: max(1, len(x_labels) // 12)],
	   ticktext=x_labels[:: max(1, len(x_labels) // 12)],
	  ),
	  yaxis=dict(
	   title="Type",
	   tickmode="array",
	   tickvals=labels2,
	   ticktext=[_shorten_label(t, max_len=28) for t in labels2],
	  ),
	  zaxis=dict(title="Value"),
	 ),
	)
	rank = np.nanmax(np.abs(mat), axis=1) if mat.size else np.array([], dtype=float)
	_add_theme_toggle(fig)
	return fig


def build_profit_3d_tickers_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	x = _to_datetime_index(r["lookup_days"])
	tickers = _as_str_list(r["tickers"])
	if tickers is None:
		return go.Figure(layout=dict(title=_safe_title(board, "PnL — 3D Surface (Tickers) — no data")))
	tp = _as_2d_float(r["tickers_profit"], shape0=len(tickers), shape1=len(x))
	if tp is None:
		return go.Figure(layout=dict(title=_safe_title(board, "PnL — 3D Surface (Tickers) — no data")))
	cash_profit_array = _as_1d_float(r.get("cash_profit_array", None), len(x))
	x_labels = [d.strftime("%Y-%m-%d") if not pd.isna(d) else "NaT" for d in x]
	if cash_profit_array is not None and len(cash_profit_array) == len(x):
		tp_all = np.vstack([tp, cash_profit_array.reshape(1, -1)])
		tickers_all = list(tickers) + ["Cash Profit"]
	else:
		tp_all = tp
		tickers_all = list(tickers)
	max_abs = np.nanmax(np.abs(tp_all), axis=1)
	show_idx = _top_n_indices(max_abs, n=min(20, len(tickers_all)))

	try:
		cash_idx = tickers_all.index("Cash Profit")
		if cash_idx not in set(show_idx.tolist()):
			show = show_idx.tolist()
			if len(show) >= 1:
				show[-1] = cash_idx
				show_idx = np.array(sorted(set(show), key=show.index), dtype=int)
	except Exception:
		pass
	tp2 = tp_all[show_idx]
	tickers2 = [tickers_all[i] for i in show_idx]
	fig = go.Figure(
	 data=[
	  go.Surface(
	   z=tp2,
	   x=x_labels,
	   y=tickers2,
	   colorscale="RdYlGn",
	   cmid=0,
	   colorbar=dict(title="PnL"),
	   hovertemplate="Ticker=%{y}<br>Date=%{x}<br>PnL=%{z:.2f}<extra></extra>",
	  )
	 ]
	)

	try:
		mean_z = np.nanmean(tp2, axis=0)
		med_z = np.nanmedian(tp2, axis=0)
		z_mean = np.tile(mean_z, (len(tickers2), 1))
		z_med = np.tile(med_z, (len(tickers2), 1))
		fig.add_trace(
		 go.Surface(
		  z=z_mean,
		  x=x_labels,
		  y=tickers2,
		  name="Mean",
		  showlegend=True,
		  showscale=False,
		  opacity=0.32,
		  colorscale="Greys",
		  visible="legendonly",
		  hovertemplate="Mean=%{z:.2f}<extra></extra>",
		 )
		)
		fig.add_trace(
		 go.Surface(
		  z=z_med,
		  x=x_labels,
		  y=tickers2,
		  name="Median",
		  showlegend=True,
		  showscale=False,
		  opacity=0.32,
		  colorscale="Greys",
		  visible="legendonly",
		  hovertemplate="Median=%{z:.2f}<extra></extra>",
		 )
		)
	except Exception:
		pass

	try:
		fig.update_layout(meta={"tc_surface_full": {"x": x_labels, "y": list(tickers_all), "z": tp_all.tolist()}})
	except Exception:
		pass
	fig.update_layout(
	 title=_safe_title(board, "Trade Calculator — PnL 3D (Tickers, top by |PnL|)"),
	 height=900,
	 scene=dict(
	  xaxis=dict(
	   title="Date",
	   tickmode="array",
	   tickvals=x_labels[:: max(1, len(x_labels) // 12)],
	   ticktext=x_labels[:: max(1, len(x_labels) // 12)],
	  ),
	  yaxis=dict(
	   title="Ticker",
	   tickmode="array",
	   tickvals=tickers2,
	   ticktext=[_shorten_label(t, max_len=24) for t in tickers2],
	  ),
	  zaxis=dict(title="PnL"),
	 ),
	)
	_add_theme_toggle(fig)
	return fig


def build_profit_3d_types_figure(results: List[Any], board: Optional[str] = None) -> go.Figure:
	r = _extract(results)
	x = _to_datetime_index(r["lookup_days"])
	types_unique = _as_str_list(r["types_unique"])
	tp = r.get("types_profits", None)
	if types_unique is None or tp is None:
		return go.Figure(layout=dict(title=_safe_title(board, "PnL — 3D Surface (Types) — no data")))
	cash_profit_array = _as_1d_float(r.get("cash_profit_array", None), len(x))
	rows = []
	labels = []
	for name, arr in zip(types_unique, list(tp)):
		a = _as_1d_float(arr, len(x))
		if a is None:
			continue
		labels.append(str(name))
		rows.append(a)
	if cash_profit_array is not None and len(cash_profit_array) == len(x):
		labels.append("Cash Profit")
		rows.append(cash_profit_array.astype(float))
	if not rows:
		return go.Figure(layout=dict(title=_safe_title(board, "PnL — 3D Surface (Types) — no data")))
	mat = np.array(rows, dtype=float)
	x_labels = [d.strftime("%Y-%m-%d") if not pd.isna(d) else "NaT" for d in x]
	labels2 = labels
	fig = go.Figure(
	 data=[
	  go.Surface(
	   z=mat,
	   x=x_labels,
	   y=labels2,
	   colorscale="RdYlGn",
	   cmid=0,
	   colorbar=dict(title="PnL"),
	   hovertemplate="Type=%{y}<br>Date=%{x}<br>PnL=%{z:.2f}<extra></extra>",
	  )
	 ]
	)

	try:
		mean_z = np.nanmean(mat, axis=0)
		med_z = np.nanmedian(mat, axis=0)
		z_mean = np.tile(mean_z, (len(labels2), 1))
		z_med = np.tile(med_z, (len(labels2), 1))
		fig.add_trace(
		 go.Surface(
		  z=z_mean,
		  x=x_labels,
		  y=labels2,
		  name="Mean",
		  showlegend=True,
		  showscale=False,
		  opacity=0.32,
		  colorscale="Greys",
		  visible="legendonly",
		  hovertemplate="Mean=%{z:.2f}<extra></extra>",
		 )
		)
		fig.add_trace(
		 go.Surface(
		  z=z_med,
		  x=x_labels,
		  y=labels2,
		  name="Median",
		  showlegend=True,
		  showscale=False,
		  opacity=0.32,
		  colorscale="Greys",
		  visible="legendonly",
		  hovertemplate="Median=%{z:.2f}<extra></extra>",
		 )
		)
	except Exception:
		pass

	try:
		fig.update_layout(meta={"tc_surface_full": {"x": x_labels, "y": labels2, "z": mat.tolist()}})
	except Exception:
		pass
	fig.update_layout(
	 title=_safe_title(board, "Trade Calculator — PnL 3D (Types)"),
	 height=850,
	 scene=dict(
	  xaxis=dict(
	   title="Date",
	   tickmode="array",
	   tickvals=x_labels[:: max(1, len(x_labels) // 12)],
	   ticktext=x_labels[:: max(1, len(x_labels) // 12)],
	  ),
	  yaxis=dict(
	   title="Type",
	   tickmode="array",
	   tickvals=labels2,
	   ticktext=[_shorten_label(t, max_len=28) for t in labels2],
	  ),
	  zaxis=dict(title="PnL"),
	 ),
	)
	rank = np.nanmax(np.abs(mat), axis=1) if mat.size else np.array([], dtype=float)
	_add_theme_toggle(fig)
	return fig


def open_visual(results: List[Any], spec: VisualSpec, board: Optional[str] = None) -> None:
	out = spec.builder(results, board)
	if isinstance(out, go.Figure):
		show_figure_in_browser(out, title=spec.label)
		return
	if isinstance(out, str):
		show_html_in_browser(out, title=spec.label)
		return
	raise TypeError(f"Unsupported visual output type: {type(out)}")

