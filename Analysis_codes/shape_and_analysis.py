import numpy as np
import os
import time
import glob
from numba import jit, njit, float64

import alphashape
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiLineString, LineString, MultiPolygon
from shapely.plotting import plot_polygon
from pathlib import Path
import re


class alpha_shape:

    def __init__(self, points, alpha_val):
        self.a_shape = alphashape.alphashape(points, alpha_val)
        self.points = points
        self.alpha_val = alpha_val


    def get_shape(self):
        # uses alpha shape to obtain alpha shape and respective area, perimeter
        
        return self.a_shape 



    def plot_alphashape(self, frame = 1, outer_points= False,  dir = Path.cwd()):
        # arguments : alpha shape, original points, total number of time frames, directory to save output files(images) 

        fig, ax = plt.subplots(constrained_layout=True)
        ax.scatter(self.points.T[0], self.points.T[1], s=5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-50,50)
        ax.set_ylim(-50,50)

        geom = self.a_shape

        if geom.is_empty:
            out_dir =  os.path.join(dir, f"tumor_alphashape_test_{frame:04d}.png")
            plt.savefig(out_dir, dpi = 100, format = "png")
            plt.close(fig)
            return None
        
        def _plot_single(poly:Polygon):
            plot_polygon(poly, ax=ax, add_points=outer_points, color='orange', alpha=self.alpha_val, linewidth=1.0)
        
        if isinstance(geom, Polygon):
            _plot_single(geom)
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                _plot_single(poly)
        else:
            try:
                for poly in geom.geoms:
                    if isinstance(poly,Polygon):
                        _plot_single(poly)
            except AttributeError:
                pass

        plot_polygon(self.a_shape, ax=ax, add_points=outer_points, facecolor='orange', alpha = self.alpha_val)
        plt.savefig(os.path.join(dir, f"tumor_alphashape_test_{frame:04d}.png"), dpi = 100, format = 'png')
        plt.close(fig)
        return None
    

    def get_boundary_length(self):
        # returns the boundary length for largest polygon structure

        if isinstance(self.a_shape, Polygon):
            boundary_length = self.a_shape.length
            
        elif isinstance(self.a_shape,MultiPolygon):
            largest_poly = max(self.a_shape.geoms, key = lambda p:p.area)
            boundary_length = largest_poly.length
        else:
            boundary_length = None
        
        return boundary_length



def get_shape_index(peri, area):
    # shape index measures closeness to circle
    # si = perimeter/ 2* (pi* area)**1/2 
    si = peri / (2* np.sqrt(np.pi * area))
    return si


def MSD(positions:np.ndarray):
    # simple msd calculations ==> |x(t) - x(0)|**2 for frame(t) and averaged over all particles for each frame i.e ensemble average
    # there is no moving tau window used here  
    msd = np.mean(np.sum((positions - positions[0,:,:][np.newaxis,:,:])**2, axis = 2), axis = 1)
    return msd


def radius_of_gyration(positions):
    # Calculates the radius of gyration
    
    N = np.shape(positions)[0]
    sum_r = np.sum(positions, axis = 0)
    sum_r2 = np.sum(np.sum(positions**2, axis=1))
    rg_2 = sum_r2/N - np.sum(sum_r**2)/N**2
    return np.sqrt(rg_2)


def center_of_mass(positions):
    # Caculates the center of mass
    
    N = np.shape(positions)[0]
    sum_r = np.sum(positions,axis = 0)
    return sum_r/N


def get_invasion_radius(positions):
    # Calculates the distane between farthest particle and center of mass
    # returns this distance  

    com = center_of_mass(positions)
    dist_com = np.sum((positions - com)**2, axis =1)**0.5
    invasion_radius = np.max(dist_com)

    return invasion_radius




