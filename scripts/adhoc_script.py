from PIL import Image

from railroad.environment.procthor.scene import ProcTHORScene
from interruption.constants import REMAPPED_SCENES_HASH, PROCTHOR_SEED
from interruption.utilities import use_remapped_scenes

remap_dir = (
    use_remapped_scenes(REMAPPED_SCENES_HASH, PROCTHOR_SEED) if REMAPPED_SCENES_HASH else None
)

scene = ProcTHORScene(seed=PROCTHOR_SEED, remove_duplicate_objects=True)

Image.fromarray(scene.get_top_down_image()).save("scene_image3.png")


print('hello')