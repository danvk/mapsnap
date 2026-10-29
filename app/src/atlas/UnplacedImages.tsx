/**
 * The images of an edition the run could not place, as thumbnails.
 *
 * For a small town of one or two sheets this is often all there is: placing a
 * sheet of a few blocks exactly is hard, and seeing it matters more than where
 * it sits, since the town's dot already says roughly where. A split sheet's
 * panels are shown as panels, cut out of the sheet image along their outline,
 * since a panel is what the run tried and failed to place. A click opens one
 * large, and the arrow keys step through the rest.
 */

import { useEffect, useState } from 'react';

import {
  chronoscopeImageUrl,
  itemViewerUrl,
  scaleForWidth,
  type UnplacedImage,
} from './itemRecord';

const THUMBNAIL_WIDTH = 80;
const THUMBNAIL_MAX_HEIGHT = 100;

interface CroppedImageProps {
  image: UnplacedImage;
  /** The largest box to fit the image in, in CSS pixels. */
  maxWidth: number;
  maxHeight: number;
}

/**
 * One image cut out of its sheet, fitted into a box.
 *
 * The CDN is a level-0 IIIF service, which serves fixed sizes of the whole
 * sheet and no regions, so the panel is cut out in the browser: the sheet image
 * is scaled, shifted so the panel's box is in view, and clipped to its outline.
 * The smallest size that is sharp at this scale is the one fetched.
 */
export function CroppedImage({
  image,
  maxWidth,
  maxHeight,
}: CroppedImageProps) {
  const { sheet, crop } = image;
  const box = crop ?? { x: 0, y: 0, width: sheet.width, height: sheet.height };
  const k = Math.min(maxWidth / box.width, maxHeight / box.height);
  const pixelRatio = window.devicePixelRatio || 1;
  const scale = scaleForWidth(sheet.width, sheet.width * k * pixelRatio);
  const clipPath = crop
    ? `polygon(${crop.ring.map(([x, y]) => `${x * k}px ${y * k}px`).join(', ')})`
    : undefined;
  return (
    <div
      className="atlas-crop"
      style={{ width: box.width * k, height: box.height * k }}
    >
      <img
        src={chronoscopeImageUrl(sheet, scale)}
        alt={image.image}
        loading="lazy"
        style={{
          width: sheet.width * k,
          height: sheet.height * k,
          left: -box.x * k,
          top: -box.y * k,
          clipPath,
        }}
      />
    </div>
  );
}

/** "p2__3" as "p2, panel 3". */
export function imageLabel(image: string): string {
  const match = /^(.*)__(\d+)$/.exec(image);
  return match ? `${match[1]}, panel ${match[2]}` : image;
}

interface LightboxProps {
  item: string;
  images: UnplacedImage[];
  index: number;
  onIndex: (index: number | null) => void;
}

/** One unplaced image, large, over the whole window. */
function Lightbox({ item, images, index, onIndex }: LightboxProps) {
  const image = images[index];
  useEffect(() => {
    function onKeydown(event: KeyboardEvent): void {
      if (event.key === 'Escape') onIndex(null);
      if (event.key === 'ArrowLeft' && index > 0) onIndex(index - 1);
      if (event.key === 'ArrowRight' && index < images.length - 1) {
        onIndex(index + 1);
      }
    }
    window.addEventListener('keydown', onKeydown);
    return () => window.removeEventListener('keydown', onKeydown);
  }, [index, images.length, onIndex]);
  if (!image) return null;
  return (
    <div
      className="atlas-lightbox"
      role="dialog"
      aria-label={imageLabel(image.image)}
      onClick={() => onIndex(null)}
    >
      <div
        className="atlas-lightbox-body"
        onClick={(event) => event.stopPropagation()}
      >
        <button
          type="button"
          className="atlas-lightbox-close"
          onClick={() => onIndex(null)}
          title="Close (Esc)"
          aria-label="Close"
        >
          ×
        </button>
        <CroppedImage
          image={image}
          maxWidth={window.innerWidth * 0.8}
          maxHeight={window.innerHeight * 0.75}
        />
        <div className="atlas-lightbox-caption">
          <button
            type="button"
            disabled={index === 0}
            onClick={() => onIndex(index - 1)}
            title="Previous (←)"
          >
            ‹
          </button>
          <span>
            <strong>{imageLabel(image.image)}</strong>
            {image.status === 'withheld'
              ? ' · placed, but withheld from the published run'
              : ' · not placed'}
            {' · '}
            {index + 1} of {images.length}
          </span>
          <a
            href={itemViewerUrl(item, image.sheet.seq)}
            target="_blank"
            rel="noreferrer"
          >
            Item viewer ↗
          </a>
          {image.sheet.loc_page && (
            <a href={image.sheet.loc_page} target="_blank" rel="noreferrer">
              loc.gov ↗
            </a>
          )}
          <button
            type="button"
            disabled={index === images.length - 1}
            onClick={() => onIndex(index + 1)}
            title="Next (→)"
          >
            ›
          </button>
        </div>
      </div>
    </div>
  );
}

interface UnplacedImagesProps {
  item: string;
  images: UnplacedImage[];
}

/** An edition's unplaced images as a grid of thumbnails, each opening large. */
export function UnplacedImages({ item, images }: UnplacedImagesProps) {
  const [open, setOpen] = useState<number | null>(null);
  // A different edition's images start closed.
  useEffect(() => setOpen(null), [item]);
  return (
    <>
      <div className="atlas-unplaced">
        {images.map((image, index) => (
          <button
            key={image.image}
            type="button"
            className="atlas-unplaced-thumb"
            onClick={() => setOpen(index)}
            title={`${imageLabel(image.image)}: ${image.status === 'withheld' ? 'withheld' : 'not placed'}`}
          >
            <CroppedImage
              image={image}
              maxWidth={THUMBNAIL_WIDTH}
              maxHeight={THUMBNAIL_MAX_HEIGHT}
            />
            <span className="atlas-unplaced-label">
              {image.image}
              {image.status === 'withheld' && ' · withheld'}
            </span>
          </button>
        ))}
      </div>
      {open !== null && (
        <Lightbox item={item} images={images} index={open} onIndex={setOpen} />
      )}
    </>
  );
}
