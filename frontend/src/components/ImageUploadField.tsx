'use client';
import { useEffect, useRef, useState } from 'react';

const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp'];
const MAX_MB = 5;

interface FileEntry {
  file: File;
  preview: string;
  error?: string;
}

interface Props {
  onChange: (files: File[]) => void;
}

export function ImageUploadField({ onChange }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [entries, setEntries] = useState<FileEntry[]>([]);
  const [dragging, setDragging] = useState(false);

  // Preview object URLs are per-file and must be released when the field
  // unmounts, or the blobs leak for the life of the tab.
  useEffect(() => {
    return () => {
      entries.forEach((e) => URL.revokeObjectURL(e.preview));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const validate = (file: File): string | undefined => {
    if (!ACCEPTED.includes(file.type)) {
      return `${file.name}: unsupported type (JPEG, PNG or WEBP only)`;
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      return `${file.name}: exceeds the ${MAX_MB} MB limit`;
    }
    return undefined;
  };

  const addFiles = (newFiles: File[]) => {
    const added: FileEntry[] = newFiles.map((file) => ({
      file,
      preview: URL.createObjectURL(file),
      error: validate(file),
    }));
    setEntries((prev) => {
      const updated = [...prev, ...added];
      onChange(updated.filter((e) => !e.error).map((e) => e.file));
      return updated;
    });
  };

  const remove = (idx: number) => {
    setEntries((prev) => {
      const target = prev[idx];
      if (target) URL.revokeObjectURL(target.preview);
      const updated = prev.filter((_, i) => i !== idx);
      onChange(updated.filter((e) => !e.error).map((e) => e.file));
      return updated;
    });
  };

  const errors = entries.filter((e) => e.error);

  return (
    <div className="upload-field">
      <button
        type="button"
        className={`dropzone${dragging ? ' is-dragging' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          addFiles(Array.from(e.dataTransfer.files));
        }}
      >
        <span className="dropzone__icon" aria-hidden="true">📸</span>
        <span className="dropzone__title">Click or drag to upload photos</span>
        <span className="dropzone__hint">JPEG, PNG or WEBP · max {MAX_MB} MB each</span>
      </button>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED.join(',')}
        multiple
        hidden
        id="image-upload-input"
        onChange={(e) => {
          addFiles(Array.from(e.target.files || []));
          e.target.value = '';
        }}
      />

      {entries.length > 0 && (
        <ul className="thumb-row upload-previews">
          {entries.map((entry, i) => (
            <li key={`${entry.file.name}-${i}`} className="upload-preview">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={entry.preview}
                alt={`Preview of ${entry.file.name}`}
                className="img-thumb"
              />
              {entry.error && (
                <span className="upload-preview__error" aria-hidden="true">
                  ✕
                </span>
              )}
              <button
                type="button"
                className="upload-preview__remove"
                onClick={() => remove(i)}
                aria-label={`Remove ${entry.file.name}`}
              >
                ✕
              </button>
              <span className="upload-preview__name">{entry.file.name}</span>
            </li>
          ))}
        </ul>
      )}

      {errors.length > 0 && (
        <div className="alert alert-error alert-sm" role="alert">
          {errors.map((e, i) => (
            <div key={i}>{e.error}</div>
          ))}
        </div>
      )}
    </div>
  );
}
