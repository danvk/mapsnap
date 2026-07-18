import { useEffect, useRef, useState } from 'react';
import { fetchNote, saveNote, type NoteContext } from '../notes/api';

interface NoteButtonProps {
  /** The page a note attaches to (from the `?files=` deep link). */
  ctx: NoteContext;
}

// Whether a keyboard event originated in a field the user is typing into.
function isTypingTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  const tag = el?.tagName;
  return (
    tag === 'INPUT' || tag === 'TEXTAREA' || el?.isContentEditable === true
  );
}

/**
 * Notebook toggle for the debugger's top nav plus its inline editor.
 *
 * Lit when the current page has a note, greyed out otherwise. Clicking it — or
 * pressing "n" — opens a small editor over the map; the note is saved to
 * `data/<volume>/artifacts/notes/<page>.json` (a blank note deletes the file).
 */
export function NoteButton(props: NoteButtonProps) {
  const { ctx } = props;
  const [note, setNote] = useState('');
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const [saving, setSaving] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // (Re)load the note whenever the page changes; ignore a stale response if the
  // page changes again before it arrives.
  useEffect(() => {
    let cancelled = false;
    setEditing(false);
    fetchNote(ctx)
      .then((text) => {
        if (!cancelled) setNote(text);
      })
      .catch(() => {
        if (!cancelled) setNote('');
      });
    return () => {
      cancelled = true;
    };
  }, [ctx.volume, ctx.page]);

  function open(): void {
    setDraft(note);
    setEditing(true);
  }

  // Open the editor on the "n" key (unless the user is typing elsewhere).
  useEffect(() => {
    function onKeydown(e: KeyboardEvent): void {
      if (e.key !== 'n' || e.metaKey || e.ctrlKey || e.altKey) return;
      if (isTypingTarget(e.target)) return;
      e.preventDefault();
      setDraft((prev) => (editing ? prev : note));
      setEditing((prev) => !prev);
    }
    window.addEventListener('keydown', onKeydown);
    return () => window.removeEventListener('keydown', onKeydown);
  }, [editing, note]);

  // Focus the textarea when the editor opens.
  useEffect(() => {
    if (editing) textareaRef.current?.focus();
  }, [editing]);

  async function save(): Promise<void> {
    setSaving(true);
    try {
      setNote(await saveNote(ctx, draft));
      setEditing(false);
    } catch (err) {
      console.error('Failed to save note:', err);
    } finally {
      setSaving(false);
    }
  }

  return (
    <span className="note-control">
      <button
        type="button"
        className={`note-toggle${note ? ' has-note' : ''}`}
        title={note ? 'Edit note (n)' : 'Add note (n)'}
        aria-label={note ? 'Edit note' : 'Add note'}
        onClick={() => (editing ? setEditing(false) : open())}
      >
        📓
      </button>
      {editing && (
        <>
          <div className="note-backdrop" onClick={() => setEditing(false)} />
          <div className="note-editor" role="dialog" aria-label="Page note">
            <div className="note-editor-title">Note · {ctx.page}</div>
            <textarea
              ref={textareaRef}
              value={draft}
              placeholder="Type a note for this page…"
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Escape') setEditing(false);
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) void save();
              }}
            />
            <div className="note-editor-actions">
              <span className="note-editor-hint">⌘↵ to save</span>
              <button
                type="button"
                onClick={() => setEditing(false)}
                disabled={saving}
              >
                Cancel
              </button>
              <button
                type="button"
                className="note-save"
                onClick={() => void save()}
                disabled={saving}
              >
                Save
              </button>
            </div>
          </div>
        </>
      )}
    </span>
  );
}
