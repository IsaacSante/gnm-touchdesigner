# Dump a TouchDesigner network to a Markdown file so it can be read outside TD:
# every operator, its wiring, non-default parameters (values, expressions,
# exports, binds), custom parameters, errors/warnings, storage, VFS files,
# and the full text of every DAT (Python, GLSL, tables, callbacks...).
#
# Run from the Textport (Dialogs > Textport) — paste this one line:
#
#   p='/Users/isaac/Documents/prototypes/human-control/gnm-touchdesigner/tools/dump_network.py'; exec(open(p).read(), dict(globals(), __file__=p))
#
# It dumps the network currently open in the Network Editor (go inside the
# GNM component first to dump just that). To pick a path explicitly, add
# DUMP_ROOT='/project1/google_gnm_model'; before exec(...).
#
# Also works pasted into a Text DAT (right-click > Run Script).
#
# Output: network_dump.md in OUT_DIR below (the repo root, gitignored).

import os

MAX_DAT_CHARS = 20000   # truncate DATs longer than this
MAX_VAL_CHARS = 200     # truncate long parameter values
SKIP = ('/sys', '/ui', '/local')
# Where network_dump.md is written (falls back to the .toe's folder if missing).
OUT_DIR = '/Users/isaac/Documents/prototypes/human-control/gnm-touchdesigner'


def _root():
	path = globals().get('DUMP_ROOT')
	if path:
		r = op(path)
		if r is None:
			raise ValueError('DUMP_ROOT not found: ' + path)
		return r
	for pane in ui.panes:
		if pane.type == PaneType.NETWORKEDITOR:
			return pane.owner
	return op('/project1') or op('/')


def _short(v, n=MAX_VAL_CHARS):
	s = repr(v) if not isinstance(v, str) else v
	s = s.replace('\n', '\\n')
	return s if len(s) <= n else s[:n] + '...'


def _par_line(p):
	mode = p.mode
	if mode == ParMode.EXPRESSION:
		desc = '= `{}`  → {}'.format(p.expr, _short(p.eval()))
	elif mode == ParMode.EXPORT:
		src = getattr(p, 'exportSource', None)
		desc = '<- export {}  → {}'.format(getattr(src, 'path', src) if src else '?', _short(p.eval()))
	elif mode == ParMode.BIND:
		desc = 'bind `{}`  → {}'.format(p.bindExpr, _short(p.eval()))
	else:
		desc = _short(p.val)
	return '- `{}` {}'.format(p.name, desc)


def _custom_par_line(p):
	bits = ['`{}`'.format(p.name), '"{}"'.format(p.label), p.style]
	if p.isNumber:
		bits.append('range {}..{}'.format(p.normMin, p.normMax))
	if p.isMenu and p.menuNames:
		bits.append('menu [{}]'.format(', '.join(p.menuNames[:20])))
	bits.append('default {}'.format(_short(p.default, 60)))
	line = '- ' + ' · '.join(str(b) for b in bits) + ' — ' + _par_line(p)[len(p.name) + 5:]
	return line


def _lang(o, text):
	try:
		lang = o.par.language.eval()
		if lang:
			return str(lang).lower()
	except Exception:
		pass
	if o.isTable:
		return 'tsv'
	if '#version' in text or 'void main' in text or 'TDOutputSwizzle' in text:
		return 'glsl'
	return 'python'


def _section(o):
	out = []
	head = '### `{}` ({})'.format(o.path, o.OPType)
	flags = [f for f, on in (('BYPASS', getattr(o, 'bypass', False)),
							  ('LOCKED', getattr(o, 'lock', False))) if on]
	if flags:
		head += '  **' + ' '.join(flags) + '**'
	out.append(head)

	ins = [i.path for i in o.inputs if i is not None]
	if ins:
		out.append('inputs: ' + ', '.join('`{}`'.format(i) for i in ins))
	if getattr(o, 'dock', None):
		out.append('docked to: `{}`'.format(o.dock.path))
	if o.comment:
		out.append('comment: ' + o.comment.replace('\n', ' '))

	try:
		errs = o.errors(recurse=False)
		warns = o.warnings(recurse=False)
	except Exception:
		errs = warns = ''
	if errs:
		out.append('**ERRORS:** ' + errs.replace('\n', ' | '))
	if warns:
		out.append('**warnings:** ' + warns.replace('\n', ' | '))

	pars = []
	for p in o.pars():
		if p.isCustom or p.readOnly:
			continue
		try:
			if p.isDefault:
				continue
		except Exception:
			pass
		try:
			pars.append(_par_line(p))
		except Exception as e:
			pars.append('- `{}` <error reading: {}>'.format(p.name, e))
	if pars:
		out.append('parameters (non-default):')
		out += pars

	if o.isCOMP:
		custom = o.customPars
		if custom:
			out.append('custom parameters ({}):'.format(len(custom)))
			page = None
			for p in custom:
				if p.page.name != page:
					page = p.page.name
					out.append('- **page: {}**'.format(page))
				try:
					out.append('  ' + _custom_par_line(p))
				except Exception as e:
					out.append('  - `{}` <error reading: {}>'.format(p.name, e))
		if o.storage:
			out.append('storage: ' + ', '.join(
				'`{}`={}'.format(k, _short(v, 80)) for k, v in o.storage.items()))
		try:
			files = list(o.vfs)
		except Exception:
			files = []
		if files:
			out.append('VFS files:')
			out += ['- `{}` ({:,} bytes)'.format(f.name, f.size) for f in files]

	if o.isDAT:
		try:
			text = o.text
		except Exception as e:
			text = '<error reading text: {}>'.format(e)
		if text.strip():
			size = '{} rows × {} cols'.format(o.numRows, o.numCols) if o.isTable else '{} lines'.format(text.count('\n') + 1)
			out.append('content ({}):'.format(size))
			if len(text) > MAX_DAT_CHARS:
				text = text[:MAX_DAT_CHARS] + '\n... [truncated, {:,} chars total]'.format(len(o.text))
			out.append('```' + _lang(o, text))
			out.append(text.rstrip('\n'))
			out.append('```')
	return '\n'.join(out)


def dump():
	root = _root()
	ops = [root] + [o for o in root.findChildren() if not o.path.startswith(SKIP)]
	ops.sort(key=lambda o: o.path)

	lines = [
		'# Network dump: `{}`'.format(root.path),
		'',
		'TouchDesigner {}.{} · {} · {} operators'.format(
			app.version, app.build, project.name, len(ops)),
		'',
		'## Overview',
		'',
	]
	base = root.path.rstrip('/').count('/')
	for o in ops:
		depth = o.path.rstrip('/').count('/') - base
		ins = [i.name for i in o.inputs if i is not None]
		err = ' ⚠️' if o.errors(recurse=False) else ''
		lines.append('{}- `{}` {}{}{}'.format(
			'  ' * depth, o.name, o.OPType,
			' ← ' + ', '.join(ins) if ins else '', err))
	lines += ['', '## Operators', '']
	for o in ops:
		lines.append(_section(o))
		lines.append('')

	out_dir = OUT_DIR if os.path.isdir(OUT_DIR) else project.folder
	out_path = os.path.join(out_dir, 'network_dump.md')
	with open(out_path, 'w', encoding='utf-8') as f:
		f.write('\n'.join(lines))
	print('Dumped {} operators from {} -> {} ({:,} bytes)'.format(
		len(ops), root.path, out_path, os.path.getsize(out_path)))


dump()
