# gen_tiles.py
from pathlib import Path
import rasterio as rio
from rasterio.windows import Window
import numpy as np
from multiprocessing import Pool
from itertools import product
import ray
from ray.cluster_utils import Cluster
from skimage.io import imsave

# Tiles are 40, 40 to make each tile represent 10km^2 since each cell is 250m^2
TILE_SIZE = (40, 40)
NUM_CPUS  = 4

def main():
	"""
	main()
	Runs a Ray cluster, then cycles through the .asc files under ../data/2020
	and their twins in ../data/2021, reads those in, turns them into numpy arrays
	and uses Ray to process all of them by passing them to process_tile
	"""
	cluster = Cluster(
		initialize_head=True,
		head_node_args={"num_cpus": NUM_CPUS})

	ray.init(address=cluster.address)
	data_dir = Path('../data/')
	years = []
	big_tile_nps = []
	for file in data_dir.iterdir():
		if file.name.endswith('.asc'):
			year = file.name.split('_')[3].split('-')[0]
			year_path = Path('../data/'+year)
			years.append(year)
			if not year_path.exists():
				year_path.mkdir()
			filename = data_dir / file.name
			with rio.open(filename) as src:
				big_tile_np = src.read(1)
			big_tile_np = np.where(big_tile_np < 0, -1, big_tile_np)
			big_tile_np = np.where(big_tile_np > 0,  1, big_tile_np)
			big_tile_nps.append(big_tile_np)
	width, height = np.shape(big_tile_nps[0])
	array_ref = ray.put(big_tile_nps)
	ray.get([process_tile.remote(prod_tup[0], prod_tup[1], years, array_ref) for prod_tup in product(range(0, width, 40), range(0,height,40))])


@ray.remote
def process_tile(i: int, j: int, years: list, big_tile_nps: list[np.array]) -> None:
	"""
	process_tile:
	A Ray remote function that takes care of subsetting, checking, and writing
	the matching tiles for index i,j and the given big_tiles.
	"""
	tiles = []
	try: 
		tile = big_tile_nps[0][i:i+TILE_SIZE[0], j:j+TILE_SIZE[1]]
		tile_sum = np.sum(tile)
		if np.all(tile==-1) or tile_sum < -1550:
			return
	except IndexError:
		return
	use_tile = False
	for big_tile_np in big_tile_nps:
		tile = big_tile_np[i:i+TILE_SIZE[0], j:j+TILE_SIZE[1]]
		tiles.append(tile)
		if tile.sum() > 20:
			use_tile = True

	all_summed = sum(tiles)
	all_summed_gt_1 = np.where(all_summed >= 1, 1, 0)
	if np.sum(all_summed_gt_1) > 80:
		use_tile = True

	if not use_tile:
		return

	for tile, year in zip(tiles, years):
		tile_f_name = '../data/'+year+'/'+str(i)+'_'+str(j)+'.tif'
		if not Path(tile_f_name).exists():
			tile_f = rio.open(tile_f_name, 
								 'w', 
								 driver='GTiff',
								 height=TILE_SIZE[0],
								 width=TILE_SIZE[1],
								 count=1,
								 dtype=str(tile.dtype),
								 transform=rio.Affine(1, 0, 0, 0, 1, 0),
								)
			tile_f.write(tile, indexes=1)
			tile_f.close()


if __name__ == '__main__':
	main()
