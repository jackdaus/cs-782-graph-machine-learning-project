# Utilities for creating subsets of COLMAP reconstructions
from pycolmap import Reconstruction
import random
import utils
import torch
from torch_geometric.data import Data
import numpy as np 

def sample_image_subsets(reconstruction: Reconstruction) -> tuple[list[int], list[int]]:
    # TODO: select different team captains based on some heuristic (e.g., 2 most distant images)
    # TODO: improve this algorithm! Right now, we can sometimes be left with images that are not added to any team. This is
    # because neither team captain can see them. We need to do something more sophisticated, like build a minimum spanning tree kinda thing

    # Create a set for all images indexes that we can draft from
    draft_pool = set(reconstruction.images.keys())

    # Sample (without replacement!) the two team captains
    captain_a_id, captain_b_id = random.sample(list(draft_pool), 2)
    # print(f"Captain A: {captain_a_id}\nCaptain B: {captain_b_id}")

    # Remove our captains from the pool
    draft_pool.remove(captain_a_id)
    draft_pool.remove(captain_b_id)

    # Create the sets that will hold our teams. Start off by adding the team captains
    team_a: set = {captain_a_id}
    team_b: set = {captain_b_id}

    # To start simple, let's just do a random drafting strategy. We might improve this later by doing something like
    # drafting nearest neighbors.
    covisible_captain_a = set(get_covisible_image_ids(reconstruction, captain_a_id))
    covisible_captain_b = set(get_covisible_image_ids(reconstruction, captain_b_id))

    # Just in case, make sure that our captains are not covisible to each other
    covisible_captain_a.discard(captain_b_id)
    covisible_captain_b.discard(captain_a_id)

    # Iterate until teams cannot draft anymore
    team_a_valid_draft_choices = covisible_captain_a & draft_pool
    team_b_valid_draft_choices = covisible_captain_b & draft_pool
    while team_a_valid_draft_choices or team_b_valid_draft_choices:
        # Make sure there are still elements covisible to captain A. Otherwise, we are done drafting for team A
        team_a_valid_draft_choices = covisible_captain_a & draft_pool
        if team_a_valid_draft_choices:
            # For now, we will just select a random covisible image. In the future, we can get fancy with nearest neighbor strategies
            draft_choice = random.sample(list(team_a_valid_draft_choices), 1)[0]
            draft_pool.remove(draft_choice)
            team_a.add(draft_choice)

        # Make sure there are still elements covisible to captain B. Otherwise, we are done drafting for team B
        team_b_valid_draft_choices = covisible_captain_b & draft_pool
        if team_b_valid_draft_choices:
            draft_choice = random.sample(list(team_b_valid_draft_choices), 1)[0]
            draft_pool.remove(draft_choice)
            team_b.add(draft_choice)

    # Sanity check: our teams should not have any shared members!
    assert team_a.isdisjoint(team_b)
    return list(team_a), list(team_b)


def get_covisible_image_ids(reconstruction: Reconstruction, image_id: int) -> list[int]:
    image = reconstruction.image(image_id)
    # Get all visible Point3D for this image
    visible_point3D_ids = [point2D.point3D_id for point2D in image.get_observation_points2D()]

    # Find all other images that can see this same Point3D
    covisible_image_ids = set()
    for point3D_id in visible_point3D_ids:
        # Get the Point3D object from the map
        point3D = reconstruction.point3D(point3D_id)
        # Get the track associated with this Point3D
        track = point3D.track
        # Each track has 2 or more elements. These elements are the other images that see this Point3D.
        for track_element in track.elements:
            covisible_image_ids.add(track_element.image_id)
    # Remove the original query imageid. We don't want this in the set!
    covisible_image_ids.remove(image_id)
    return list(covisible_image_ids)


def reconstruction_to_pyg_data(reconstruction: Reconstruction, filtered_image_ids: list[int] = None) -> Data:
    # Get all the image_ids from this Reconstruction
    all_image_ids = list(reconstruction.images.keys())

    # If no image_ids were provided, then we use all images in the Reconstruction
    if filtered_image_ids is None:
        filtered_image_ids = all_image_ids # TODO make sure this not a reference type!

    assert set(filtered_image_ids).issubset(set(all_image_ids))

    # Calculate per node features. For now, we will just use the image positions as our node features 
    image_centers = [image.projection_center() for image in reconstruction.images.values() if image.image_id in filtered_image_ids]
    # x = torch.tensor(image_centers, dtype=torch.float)
    x = torch.from_numpy(np.array(image_centers)).float()
    
    # Create a map between the image_id and index in the list
    image_id_to_idx = {image_id: idx for idx, image_id in enumerate(filtered_image_ids)}

    # Initialize our lists for the edges. Source node and destination node. 
    # (The graph is undirected, so "src" and "dst" labels are just for convention here.)
    src, dst = [], []
    # For each image in set A...
    for src_id in filtered_image_ids:
        # Get all other images that have tracks linked to the source image
        for dst_id in utils.subset.get_covisible_image_ids(reconstruction, src_id):
            # We want to make sure the 
            if dst_id in image_id_to_idx:
                src.append(image_id_to_idx[src_id])
                dst.append(image_id_to_idx[dst_id])
    # Create the edge index matrix. PyG expects a matrix of shape (2, E) 
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index)
    return data

