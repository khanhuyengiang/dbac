"""
plot_bloch_sphere.py

A utility module for visualizing quantum states on the Bloch sphere. 
Includes functions to convert quantum states to Bloch vector coordinates, 
plot the Bloch sphere, render Bloch vectors, and highlight 3D outlines.

Functions:
- density_matrix_to_bloch: Converts a density matrix or statevector to Bloch vector coordinates.
- plot_bloch_sphere: Draws the Bloch sphere with key reference lines and labels.
- plot_bloch_vector: Renders a Bloch vector and optional label on the sphere.
- add_3d_outline: Adds a silhouette outline to a 3D surface for visual clarity.

Dependencies:
- numpy
- matplotlib
"""

__all__ = [
    "density_matrix_to_bloch",
    "plot_bloch_sphere",
    "plot_bloch_vector",
    "add_3d_outline",
    "plot_bloch_trajectory",
]


import numpy as np
import matplotlib.pyplot as plt

def density_matrix_to_bloch(state):
    """
    Convert a 2x2 density matrix to Bloch vector coordinates (u, v, w).
    
    Parameters:
    density_matrix (ndarray): A 2x2 density matrix.
    
    Returns:
    tuple: The Bloch vector components (u, v, w).
    """
    if np.shape(state)[0] != np.shape(state)[1]:
        density_matrix = np.outer(state, np.conj(state))
    elif np.shape(state)[0] == np.shape(state)[1]:
        density_matrix = state

    # Extract matrix elements
    rho_00 = density_matrix[0, 0]
    rho_01 = density_matrix[0, 1]
    rho_10 = density_matrix[1, 0]
    rho_11 = density_matrix[1, 1]
    
    # Calculate the components of the Bloch vector
    u = 2 * np.real(rho_01)
    v = 2 * np.imag(rho_10)
    w = rho_00 - rho_11
    
    return u, v, w

# Function to plot the Bloch vector on the Bloch sphere
def plot_bloch_sphere(font_size=20):
    """
    Plots the Bloch sphere with the surface, latitude and longitude lines, and the |0⟩ and |1⟩ labels at the poles.
    
    Parameters:
    ax (matplotlib.axes._axes.Axes): The axes on which to plot the Bloch sphere.
    font_size (int): Font size for labels.
    """
    # Create a meshgrid for the sphere surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    u, v = np.meshgrid(u, v)

    # Parametric equations for the sphere surface
    x_sphere = np.sin(v) * np.cos(u)
    y_sphere = np.sin(v) * np.sin(u)
    z_sphere = np.cos(v)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='b',alpha=0.03)
    
    # Plot the longitude lines (azimuthal angles)
    for i in range(0, 360, 90):  # Longitude lines at every 90 degrees
        phi = np.radians(i)
        x_longitude = np.sin(v) * np.cos(phi)
        y_longitude = np.sin(v) * np.sin(phi)
        z_longitude = np.cos(v)
        ax.plot(x_longitude, y_longitude, z_longitude, color='k', lw=0.4, alpha=0.2)
    
    # Plot the latitude lines (polar angles)
    # for i in range(0, 180, 90):  # Latitude lines at every 90 degrees
    theta = np.radians(90)
    x_latitude = np.outer(np.cos(u), np.sin(theta))
    y_latitude = np.outer(np.sin(u), np.sin(theta))
    z_latitude = np.outer(np.ones(np.size(u)), np.cos(theta))
    ax.plot(x_latitude, y_latitude, z_latitude, color='k', lw=0.1, alpha=0.01)

    # Add the |0⟩ label at the north pole
    ax.text(0, 0, 1.2, r'$|0\rangle$', fontsize=font_size, color='k', ha='center', va='center')

    # Add the |1⟩ label at the south pole
    ax.text(0, 0, -1.1, r'$|1\rangle$', fontsize=font_size, color='k', ha='center', va='center')

    # Plot the x, y, z axes as black thick lines
    ax.plot([0, 1], [0, 0], [0, 0], color='k', lw=1)  # X axis
    ax.plot([0, 0], [0, 1], [0, 0], color='k', lw=1)  # Y axis
    ax.plot([0, 0], [0, 0], [0, 1], color='k', lw=1)  # Z axis

    # Add labels at the end of each axis
    ax.text(1.1, 0, 0, r'$x$', fontsize=font_size, color='k', ha='center', va='center')
    ax.text(0, 1.1, 0, r'$y$', fontsize=font_size, color='k', ha='center', va='center')
    ax.text(0, 0, 1.1, r'$z$', fontsize=font_size, color='k', ha='center', va='center')

    # Set the view angle and axis limits
    # ax.set_title("Bloch Sphere Representation", fontsize=font_size)
    ax.view_init(30, 30)  # Adjust view angle to look like a globe
    max_range = 1  # Since the Bloch sphere is a unit sphere
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_box_aspect([1, 1, 1])  # Equal scaling in all axes
    ax.set_axis_off()  # Hide the axes
    ax.grid(False)  # Remove the grid
    ax.set_facecolor('none')  # Remove the box/frame
    plt.tight_layout()

    return ax

def plot_bloch_vector(state, ax, color='r', num_points=100, text=None, marker=None, marker_color='k'):
    """
    Plots the Bloch vector as a red arrow on the Bloch sphere along with a latitude circle, 
    and optionally adds a text label to the latitude circle.

    Parameters:
    state (ndarray): The density matrix or state vector (2x2) for the system.
    ax (matplotlib.axes._axes.Axes): The axes on which to plot the vector.
    color (str): The color of the Bloch vector and latitude circle.
    num_points (int): Number of points to plot for the latitude circle.
    text (str or None): Text to place on the latitude circle (default is None).
    """
    # Get the Bloch vector components (x, y, z) from the density matrix or statevector
    x, y, z = density_matrix_to_bloch(state)

    # Plot the Bloch vector (the red arrow)
    ax.quiver(0, 0, 0, x, y, z, color=color, length=1.0, arrow_length_ratio=0.1)

    if marker != None:
        # Plot a marker at the tip of the Bloch vector (x, y, z)
        ax.plot(x, y, z, color=marker_color, marker=marker,markersize=15)  # s is the size of the marker


    # Latitude circle components
    theta = np.arccos(np.real(z))
    u = np.linspace(0, 2 * np.pi, num_points)
    x_latitude = np.outer(np.cos(u), np.sin(theta))
    y_latitude = np.outer(np.sin(u), np.sin(theta))
    z_latitude = np.outer(np.ones(np.size(u)), np.cos(theta))
    ax.plot(x_latitude, y_latitude, z_latitude, color=color, lw=0.7, alpha=0.5)

    # Add text to the latitude circle if provided
    if text:
        # Example: Placing text at a specific angle on the latitude circle (at angle=pi/4)
        angle = 0.15  # You can adjust this angle as needed
        x_text = np.sin(theta) * np.cos(angle)
        y_text = np.sin(theta) * np.sin(angle)
        z_text = np.cos(theta)
        ax.text(x_text, y_text, z_text, text, color=color, fontsize=20, ha='center', va='bottom')

    return ax

def add_3d_outline(ax, x, y, z, color='black', linewidth=2):
    """
    Adds a silhouette outline to a 3D surface by detecting edge points.

    Parameters:
    ax : Matplotlib 3D axis
    x, y, z : 2D arrays defining the surface
    color : str, outline color
    linewidth : int, thickness of the outline

    Returns:
    ax : Updated 3D axis with the outline
    """

    # Compute surface normals using cross product
    dx, dy = np.gradient(x), np.gradient(y)
    dz = np.gradient(z)
    normals = np.cross(np.c_[dx[1], dy[1], dz[1]], np.c_[dx[0], dy[0], dz[0]])

    # Normalize normals
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    # Viewing direction (assume camera at (0,0,10), looking at origin)
    view_dir = np.array([0, 0, 10]) - np.array([0, 0, 0])
    view_dir /= np.linalg.norm(view_dir)

    # Compute dot product: If close to zero, it's an edge
    dot_product = np.abs(np.dot(normals, view_dir))
    edge_mask = dot_product < 0.2  # Threshold for edge detection

    # Extract silhouette points
    edge_x, edge_y, edge_z = x[edge_mask], y[edge_mask], z[edge_mask]

    # Plot the detected silhouette
    ax.plot(edge_x, edge_y, edge_z, color=color, linewidth=linewidth)

    return ax


def plot_bloch_trajectory(states, texts, *, colors=None, marker=None, marker_color=None,
                           target_fidelity=None, save_path=None):
    """
    Plot a sequence of states (e.g. the DB-QITE recursion states returned by
    `GCI.execute_gci`) on the Bloch sphere, one latitude circle per state, and
    optionally a dashed latitude circle marking a target fidelity with |0>.

    Parameters:
        states (sequence): Density matrices or state vectors, one per step.
        texts (sequence[str]): Label for each state's latitude circle.
        colors (sequence[str], optional): Color per state. Defaults to
            ['C0', '#2c0084', '#7a0085', '#006f7a'].
        marker (str, optional): Marker for the first (initial) state.
        marker_color (str, optional): Marker color for the first state.
        target_fidelity (float, optional): If given, draws a dashed latitude
            circle at this fidelity with |0>.
        save_path (str, optional): If given, saves the figure to this path.

    Returns:
        matplotlib.axes._axes.Axes: The 3D axes with the trajectory drawn.
    """
    if colors is None:
        colors = ['C0', '#2c0084', '#7a0085', '#006f7a']

    ax = plot_bloch_sphere()

    if target_fidelity is not None:
        theta = 2 * np.arccos(np.sqrt(target_fidelity))
        u = np.linspace(0, 2 * np.pi, 100)
        x_lat = np.outer(np.cos(u), np.sin(theta))
        y_lat = np.outer(np.sin(u), np.sin(theta))
        z_lat = np.outer(np.ones(np.size(u)), np.cos(theta))
        ax.plot(x_lat, y_lat, z_lat, color='k', lw=1, alpha=1, linestyle='--')

    for i, (state, text) in enumerate(zip(states, texts)):
        if i == 0 and marker is not None:
            ax = plot_bloch_vector(state, ax, color=colors[i], text=text,
                                    marker=marker, marker_color=marker_color)
        else:
            ax = plot_bloch_vector(state, ax, color=colors[i], text=text)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=-1.1)

    return ax
