# Shard-list parsing, shared by launch.sh and supervise.sh.
#
# A fleet split across regions gives each region a slice of one partition, so
# both scripts need to name a set of shards rather than a single one: the
# instances are only disjoint if every region launches with the same --shards.

# Expand "0-3", "4,5" or "0,2-4" into one shard number per line, rejecting
# anything outside 0..shards-1 so a typo cannot silently launch nothing.
expand_shards() {
  local spec=$1 shards=$2 part start end n
  local IFS=,
  for part in $spec; do
    case "$part" in
      *-*)
        start=${part%%-*}
        end=${part##*-}
        ;;
      *) start=$part; end=$part ;;
    esac
    case "$start$end" in
      *[!0-9]*|"") echo "bad shard spec: $part" >&2; return 1 ;;
    esac
    if [ "$start" -gt "$end" ] || [ "$end" -ge "$shards" ]; then
      echo "shard spec $part is outside 0-$((shards - 1))" >&2
      return 1
    fi
    for n in $(seq "$start" "$end"); do echo "$n"; done
  done
}
