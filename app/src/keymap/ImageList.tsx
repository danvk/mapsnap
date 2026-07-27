import type { ImageInfo } from './types';

interface ImageListProps {
  images: ImageInfo[];
  selectedName: string | null;
  onSelect: (name: string) => void;
  /** Column heading; the adjacency labeler reuses this list for volume pages. */
  heading?: string;
}

/** Left-column list of available key map pages, with their label counts. */
export function ImageList(props: ImageListProps) {
  const { images, selectedName, onSelect, heading = 'Key maps' } = props;
  return (
    <div className="image-list">
      <h2>{heading}</h2>
      <ul>
        {images.map((info) => (
          <li
            key={info.name}
            className={info.name === selectedName ? 'selected' : undefined}
            onClick={() => onSelect(info.name)}
            title={
              info.hasKeymap === false
                ? `${info.name} (no keymap.json; listed because it has truth labels)`
                : info.name
            }
          >
            <span
              className={
                info.hasKeymap === false ? 'image-name no-keymap' : 'image-name'
              }
            >
              {info.name}
            </span>
            {info.withText > 0 && (
              <span className="label-count" title="labels with text">
                {info.withText}
              </span>
            )}
            {info.withoutText > 0 && (
              <span
                className="label-count label-count-empty"
                title="labels without text"
              >
                {info.withoutText}
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
