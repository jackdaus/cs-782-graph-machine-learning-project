# Utilities for creating subsets of COLMAP reconstructions

from pycolmap import Reconstruction

# Let's turn that into a function
def get_covisible_image_ids(reconstruction: Reconstruction, image_id: int):
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
