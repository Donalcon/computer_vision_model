def calculate_bbox_centre(coordinates):
    # Extract x and y coordinates from the input list
    x_coordinates = coordinates[0::2]  # Extract all even-indexed values (x coordinates)
    y_coordinates = coordinates[1::2]  # Extract all odd-indexed values (y coordinates)

    # Calculate the center of the bounding box
    Cx = sum(x_coordinates) / 4
    Cy = sum(y_coordinates) / 4

    return Cx, Cy
