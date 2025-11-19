set -euo pipefail

cd "$HOME/scratch/lscai-layer-norm"

# ? should we use some more specific base image name instead of my_env? xD
echo "Building image"
podman build -t my_env .

echo "Exporting to my_pytorch.sqsh..."
SQSH="$HOME/scratch/my_pytorch.sqsh"

echo "Running enroot import..."

# apparently enroot import returns non-zero even if it succeeds
set +e
enroot import -o my_pytorch.sqsh podman://my_env
st=$?
set -e

# Accept non-zero exit if the file exists
[[ -f my_pytorch.sqsh ]] || exit $st


# ? also here, I think we may be overriding the .sqsh that we created
# ? during the course, we migth want to change name to avoid confusions D:
# ? for now I'll keep it as is.

echo "Moving my_pytorch.sqsh to $SQSH..."
mv my_pytorch.sqsh "$SQSH"

echo "Listing $SQSH details:"
ls -lh "$SQSH"

echo "Done building and exporting image."
