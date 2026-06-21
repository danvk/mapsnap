import type { ImageInfo } from './types';

interface ImageListProps {
  images: ImageInfo[];
  selectedName: string | null;
  onSelect: (name: string) => void;
}

/** Left-column list of available key map images, with their label counts. */
export function ImageList(props: ImageListProps) {
  const { images, selectedName, onSelect } = props;
  return (
    <div className="image-list">
      <h2>Key maps</h2>
      <ul>
        {images.map((info) => (
          <li
            key={info.name}
            className={info.name === selectedName ? 'selected' : undefined}
            onClick={() => onSelect(info.name)}
          >
            <span className="image-name">{info.name}</span>
            {info.labelCount !== null && (
              <span className="label-count">{info.labelCount}</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
