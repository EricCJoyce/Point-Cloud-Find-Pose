#  Eric Joyce, Stevens Institute of Technology, 2019

#  Given an image, a PLY point cloud, and a camera matrix file, estimate where in the point cloud
#  the picture of the object seen in the image was taken.

#  python point.py valve.portrait.jpeg valve.ply -K Apple-iPhoneXS-P4.2mm.mat -v -showFeatures -o K -o Ext -o P

import sys
import re
import os
import subprocess
import numpy as np													#  Always necessary
from numpy.linalg import inv										#  Matrix inverter
import cv2															#  Core computer vision engine
import matplotlib.pyplot as plt										#  Display and save intermediate images
from open3d import *

#   argv[0] = point.py
#   argv[1] = query image (photo)
#   argv[2] = point-cloud file
#  {argv[3..n] = flags}

def main():
	if len(sys.argv) < 3:  ##########################################  Step 1: check arguments and files
		usage()
		return
	if not os.path.exists(sys.argv[1]):								#  Must have a query image
		print('Unable to find query image "' + sys.argv[1] + '"')
		return
	if not os.path.exists(sys.argv[2]):								#  Must have a 3D file
		print('Unable to find point cloud file "' + sys.argv[2] + '"')
		return

	params = parseRunParameters()									#  Get command-line options
	if params['helpme']:											#  Did user ask for help?
		usage()														#  Display options
		return

	if not os.path.exists(params['Kmat']):							#  Must have a camera intrinsics file
		print('Unable to find camera intrinsics file "' + params['Kmat'] + '"')
		return

	params['photo'] = sys.argv[1]									#  Add required arguments to the parameter dictionary
	params['cloud'] = sys.argv[2]									#  so they can get conveniently passed around
	params['cloudpath'] = '/'.join(sys.argv[2].split('/')[:-1])

	if params['verbose']:											#  Give a run-down of this script-call's settings
		print('>>> With your help, we will estimate the position of the camera that took "'+params['photo']+'" in the 3D point cloud "'+params['cloud']+'".')
		print('>>> The scene was photographed by a camera described in "'+params['Kmat']+'".')
		for outputformat in params['output']:
			if outputformat == 'rt':
				print('>>> We will write OpenCV\'s rotation and translation 3-vectors to the "point.log" file.')
			elif outputformat == 'Rt':
				print('>>> We will write the rotation matrix and (camera center) translation vector to the "point.log" file.')
			elif outputformat == 'K':
				print('>>> We will write the intrinsic matrix to the "point.log" file.')
			elif outputformat == 'Ext':
				print('>>> We will write the extrinsic matrix to the "point.log" file.')
			elif outputformat == 'P':
				print('>>> We will write the projection matrix to the "point.log" file.')
			elif outputformat == 'Meshlab':
				print('>>> We will write Meshlab pose code to the "point.log" file.')
			elif outputformat == 'Blender':
				print('>>> We will write Blender pose code to the "point.log" file.')

	photo = cv2.imread(params['photo'], cv2.IMREAD_COLOR)			#  Read photo, really just to get the width and height
	params['Kmat'] = loadKFromFile(params['Kmat'], photo)			#  Build and store intrinsic matrix from given file,
																	#  but override these if necessary from the image itself.
																	#  The image is likely to have been scaled.

	#################################################################  Step 2: set about collecting correspondences. In a
	#  dense point-cloud, this can be memory-intensive and time-consuming. That's why the first thing is to find out how much
	#  work we actually need to do. In the worst-case scenario, we must open an interactive Open3D window in which we position
	#  our virtual camera in the point cloud and let the user set up a position. From this, we get user-clicked interest points
	#  in the 3D rendering and match these to corresponding click-points in the photograph. The ultimate aim of Step 2 is to put
	#  at least four values into the list 'correspondences2_3'. Best-case scenario, we can simply retrieve contents for
	#  'correspondences2_3' from file--meaning somebody previously clicked 2D-points and their matching 3D points and wrote these
	#  to 'correspondences.txt'.
	correspondences2_3 = []											#  Will ultimately be a list of tuples of tuples:
																	#  ( (x-photo, y-photo), (x-3D, y-3D, z-3D) )

	if os.path.exists('correspondences.txt'):						#  Best-case: we already have 2D-3D correspondences, skip ahead
		fh = open('correspondences.txt', 'r')						#  Load 2D-3D correspondences and jump directly
		lines = fh.readlines()										#  to pose-estimation.
		fh.close()
		for line in lines[1:]:
			arr = line.strip().split()
			correspondences2_3.append( ((float(arr[0]), float(arr[1])), (float(arr[2]), float(arr[3]), float(arr[4]))) )
	else:															#  Otherwise, we must get 2D-3D correspondences
		pts2D = []													#  To become points in the photo
		pts3D = []													#  To become points in the RGB rendering
		virtCamExt = None

		if os.path.exists('2D.txt'):								#  2D points are (x, y) clicked in the photo
			fh = open('2D.txt', 'r')
			lines = fh.readlines()
			fh.close()
			for line in lines[1:]:									#  First line is the name of the image
				arr = line.strip().split()
				pts2D.append( (float(arr[0]), float(arr[1])) )

		if os.path.exists('3D.txt'):								#  3D points are (x, y, z) derived from rendering clicks
			fh = open('3D.txt', 'r')								#  and a virtual Open3D camera extrinsic matrix
			lines = fh.readlines()
			fh.close()
			for line in lines[1:]:									#  First line is the name of the point cloud
				arr = line.strip().split()
				pts3D.append( (float(arr[0]), float(arr[1]), float(arr[2])) )

		if os.path.exists('ext.mat'):								#  The extrinsic matrix of an interactive virtual camera
			virtCamExt = []
			fh = open('ext.mat', 'r')
			lines = fh.readlines()
			fh.close()
			for line in lines:
				arr = line.strip().split()
				virtCamExt.append( [ float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]) ] )
			virtCamExt = np.array(virtCamExt)						#  Convert list of lists to 4x4 matrix

																	#  If we have pts2D and pts3D, proceed to building
		if len(pts2D) > 0 and len(pts3D) > 0:						#  correspondences2_3; we don't care about 'virtCamExt'.
			for i in range(0, min(len(pts2D), len(pts3D))):			#  Lists should have same length, but if they don't
																	#  be sure not to reach beyond the smaller one.
				correspondences2_3.append( ((pts2D[i][0], pts2D[i][1]), (pts3D[i][0], pts3D[i][1], pts3D[i][2])) )
		else:														#  One of the two, or both are missing.
			if len(pts3D) == 0:										#  We have no 3D points
																	#  Get extrinsics (also create rgb.png and depth.npy)
				virtCamIntr, virtCamExt = open3D_virtual_camera_window(photo.shape[1], photo.shape[0], params)
				pts3D = clicker_RGB(virtCamIntr, virtCamExt, params)#  Get 3D points using virtCamExt, rgb.png, and depth.npy.
				if pts3D is None:									#  Hard-quit if the return value is None
					return

			if len(pts2D) == 0:										#  We have no photo click-points
				pts2D = clicker_photo(params)						#  Get 2D points using the clicker

																	#  Now... assemble correspondences!
			for i in range(0, min(len(pts2D), len(pts3D))):			#  Lists should have same length, but if they don't
																	#  be sure not to reach beyond the smaller one.
				correspondences2_3.append( ((pts2D[i][0], pts2D[i][1]), (pts3D[i][0], pts3D[i][1], pts3D[i][2])) )

	#################################################################  Step 3: an intermediate, auxiliary step.
	if params['showFeatures']:										#  Save 3D points to file, all bright green
		fstr  = 'ply\n'
		fstr += 'format ascii 1.0\n'
		fstr += 'comment https://github.com/EricCJoyce\n'
		fstr += 'element vertex '+str(len(correspondences2_3))+'\n'
		fstr += 'property float x\n'
		fstr += 'property float y\n'
		fstr += 'property float z\n'
		fstr += 'property uchar red\n'
		fstr += 'property uchar green\n'
		fstr += 'property uchar blue\n'
		fstr += 'end_header\n'
		for corr in correspondences2_3:								#  Write the 3D part (index [1]) of each
			fstr += str(corr[1][0])+' '+str(corr[1][1])+' '+str(corr[1][2])+' 0 255 0\n'
		fh = open('features.ply', 'w')
		fh.write(fstr)
		fh.close()

	#################################################################  Step 4: Run the PnP-solver to come up with our
																	#          pose estimate
	distortionCoeffs = np.zeros((4, 1))								#  Assume no lens distortion.

																	#  Data-type counts! Convert these explicitly!
	corr3 = np.array([ [ correspondences2_3[x][1][0], \
	                     correspondences2_3[x][1][1], \
	                     correspondences2_3[x][1][2] ] for x in range(0, len(correspondences2_3)) ], dtype=np.float64)
	corr2 = np.array([ [ correspondences2_3[x][0][0], \
	                     correspondences2_3[x][0][1] ] for x in range(0, len(correspondences2_3)) ], dtype=np.float64)

																	#  Run solver
	success, rotation, translation = cv2.solvePnP(corr3, corr2, \
	                                              params['Kmat'], distortionCoeffs, \
	                                              flags=cv2.SOLVEPNP_ITERATIVE)

	print('r\n'+str(rotation))										#  Always print to screen
	print('t\n'+str(translation))

	#################################################################  Step 5: Pack up output

	if len(params['output']) > 0:									#  At least one output format was specified
		fstr = ''													#  Prepare string to write to file
		if 'rt' in params['output']:								#  OpenCV rotation vector and
			if params['verbose']:									#  OpenCV translation vector
				print('Writing OpenCV rotation vector and translation vector to file')
			fstr += 'r: '+str(rotation[0][0])+' '+str(rotation[1][0])+' '+str(rotation[2][0])+'\n'
			fstr += 't: '+str(translation[0][0])+' '+str(translation[1][0])+' '+str(translation[2][0])+'\n\n'
		if 'Rt' in params['output']:								#  Rotation matrix and
			if params['verbose']:									#  camera-center translation vector
				print('Writing rotation matrix and (camera center) translation vector to file')
			RotMat, _ = cv2.Rodrigues(rotation)
			fstr += 'R:\n'
			fstr += str(RotMat[0][0])+' '+str(RotMat[0][1])+' '+str(RotMat[0][2])+'\n'
			fstr += str(RotMat[1][0])+' '+str(RotMat[1][1])+' '+str(RotMat[1][2])+'\n'
			fstr += str(RotMat[2][0])+' '+str(RotMat[2][1])+' '+str(RotMat[2][2])+'\n'
			t = -RotMat.dot(translation)
			fstr += 't: '+str(t[0][0])+' '+str(t[1][0])+' '+str(t[2][0])+'\n\n'
		if 'K' in params['output']:									#  Intrinsic matrix (3x3)
			if params['verbose']:
				print('Writing intrinsic matrix to file')
			fstr += 'K:\n'
			fstr += str(params['Kmat'][0][0])+' '+str(params['Kmat'][0][1])+' '+str(params['Kmat'][0][2])+'\n'
			fstr += str(params['Kmat'][1][0])+' '+str(params['Kmat'][1][1])+' '+str(params['Kmat'][1][2])+'\n'
			fstr += str(params['Kmat'][2][0])+' '+str(params['Kmat'][2][1])+' '+str(params['Kmat'][2][2])+'\n\n'
		if 'Ext' in params['output']:								#  Extrinsic matrix (3x4)
			if params['verbose']:
				print('Writing extrinsic matrix to file')
			RotMat, _ = cv2.Rodrigues(rotation)
			fstr += 'Ext:\n'
			fstr += str(RotMat[0][0])+' '+str(RotMat[0][1])+' '+str(RotMat[0][2])+' '+str(translation[0][0])+'\n'
			fstr += str(RotMat[1][0])+' '+str(RotMat[1][1])+' '+str(RotMat[1][2])+' '+str(translation[1][0])+'\n'
			fstr += str(RotMat[2][0])+' '+str(RotMat[2][1])+' '+str(RotMat[2][2])+' '+str(translation[2][0])+'\n\n'
		if 'P' in params['output']:									#  Projection matrix = Intrinsic (3x3) * Extrinsic (3x4)
			if params['verbose']:
				print('Writing projection matrix to file')
			RotMat, _ = cv2.Rodrigues(rotation)
			Ext = np.array( [ [RotMat[0][0], RotMat[0][1], RotMat[0][2], translation[0][0]], \
			                  [RotMat[1][0], RotMat[1][1], RotMat[1][2], translation[1][0]], \
			                  [RotMat[2][0], RotMat[2][1], RotMat[2][2], translation[2][0]] ] )
			P = params['Kmat'].dot(Ext)
			fstr += 'P:\n'
			fstr += str(P[0][0])+' '+str(P[0][1])+' '+str(P[0][2])+' '+str(P[0][3])+'\n'
			fstr += str(P[1][0])+' '+str(P[1][1])+' '+str(P[1][2])+' '+str(P[1][3])+'\n'
			fstr += str(P[2][0])+' '+str(P[2][1])+' '+str(P[2][2])+' '+str(P[2][3])+'\n\n'
		'''
		if 'Meshlab' in params['output']:							#  Meshlab pasty
			if params['verbose']:
				print('Writing Meshlab markup to file')
			RotMat, _ = cv2.Rodrigues(rotation)
			t = -RotMat.dot(translation)
			#  https://sourceforge.net/p/meshlab/discussion/499533/thread/cc40efe0/
			mmPerPixelx = 0.0369161
			mmPerPixely = 0.0369161
			fstr += '<!DOCTYPE ViewState>'
			fstr += '<project>'
																	#  Translation is a column vector:
			fstr += '<VCGCamera '									#  must be indexed as 2D, row, col
			fstr += 'TranslationVector="'+str(t[0][0])+' '+str(t[1][0])+' '+str(t[2][0])+' 1" '
			fstr += 'LensDistortion="0 0" '
			fstr += 'ViewportPx="'+str(params['Kmat'][0][2] * 2.0)+' '+str(params['Kmat'][1][2] * 2.0)+'" '
			fstr += 'PixelSizeMm="'+str(mmPerPixelx)+' '+str(mmPerPixely)+'" '
			fstr += 'CenterPx="'+str(int(round(params['Kmat'][0][2])))+' '+str(int(round(params['Kmat'][1][2])))+'" '
			fstr += 'FocalMm="30.00" '
			fstr += 'RotationMatrix="'+str(RotMat[0][0])+' '+str(RotMat[1][0])+' '+str(RotMat[2][0])+' 0 '
			fstr +=                    str(RotMat[0][1])+' '+str(RotMat[1][1])+' '+str(RotMat[2][1])+' 0 '
			fstr +=                    str(RotMat[0][2])+' '+str(RotMat[1][2])+' '+str(RotMat[2][2])+' 0 '
			fstr +=                   '0 0 0 1 "/>'
			fstr += '<ViewSettings NearPlane="0.909327" TrackScale="1.73205" FarPlane="8.65447"/>'
			fstr += '</project>\n\n'
		if 'Blender' in params['output']:							#  Pastable code for Blender's Python console
			if params['verbose']:
				print('Writing Blender code to file')
			rotMat, _ = cv2.Rodrigues(rotation)
			t = -RotMat.dot(translation)							#  Camera position
																	#  Set rotation in Quaternions
			RotMat = RotMat.T										#  Transpose rotation matrix
			tr = RotMat.trace()										#  Rotation matrix trace
			if tr > 0:
				sq = np.sqrt(1.0 + tr) * 2.0
				qw = 0.25 * sq
				qx = (RotMat[2][1] - RotMat[1][2]) / sq
				qy = (RotMat[0][2] - RotMat[2][0]) / sq
				qz = (RotMat[1][0] - RotMat[0][1]) / sq
			elif RotMat[0][0] > RotMat[1][1] and RotMat[0][0] > RotMat[2][2]:
				sq = np.sqrt(1.0 + RotMat[0][0] - RotMat[1][1] - RotMat[2][2]) * 2.0
				qw = (RotMat[2][1] - RotMat[1][2]) / sq
				qx = 0.25 * sq
				qy = (RotMat[0][1] + RotMat[1][0]) / sq
				qz = (RotMat[0][2] + RotMat[2][0]) / sq
			elif RotMat[1][1] > RotMat[2][2]:
				sq = np.sqrt(1.0 + RotMat[1][1] - RotMat[0][0] - RotMat[2][2]) * 2.0
				qw = (RotMat[0][2] - RotMat[2][0]) / sq
				qx = (RotMat[0][1] + RotMat[1][0]) / sq
				qy = 0.25 * sq
				qz = (RotMat[1][2] + RotMat[2][1]) / sq
			else:
				sq = np.sqrt(1.0 + RotMat[2][2] - RotMat[0][0] - RotMat[1][1]) * 2.0
				qw = (RotMat[1][0] - RotMat[0][1]) / sq
				qx = (RotMat[0][2] + RotMat[2][0]) / sq
				qy = (RotMat[1][2] + RotMat[2][1]) / sq
				qz = 0.25 * sq
  			fstr += 'import bpy\n'
			fstr += 'pi = 3.14159265\n'
			fstr += 'scene = bpy.data.scenes["Scene"]\n'
			fstr += 'scene.camera.location.x = '+str(t[0][0])+'\n'
			fstr += 'scene.camera.location.y = '+str(t[1][0])+'\n'
			fstr += 'scene.camera.location.z = '+str(t[2][0])+'\n'
			fstr += 'scene.camera.rotation_mode = "QUATERNION"\n'	#  Blender quaternions read W, X, Y, Z
																	#  Quirks of the Blender coordinate system:
																	#  Swap W and Y; negate Y.
			fstr += 'scene.camera.rotation_quaternion[0] = '+str(-qy)+'\n'
			fstr += 'scene.camera.rotation_quaternion[1] = '+str(qz)+'\n'
																	#  Swap X and Z; negate X.
			fstr += 'scene.camera.rotation_quaternion[2] = '+str(qw)+'\n'
			fstr += 'scene.camera.rotation_quaternion[3] = '+str(-qx)+'\n'
			fstr += '\n'
		'''

		fh = open('point.log', 'w')									#  Write to file
		fh.write(fstr)
		fh.close()

		write_camera_pyramid(rotation, translation, params)			#  Create a PLY file of the camera that took
																	#  the given photo
	return

#  Opens an interactive Open3D window in which we set up a camera pose.
#  This function (should) creates the rendering file, 'rgb.png' (see comment, lines XXX-XXX)
#  and the Z buffer 'depth.npy'.
#  Write the virtual camera extrinsic matrix to file 'ext.mat'.
#  Write the virtual camera intrinsic matrix to file 'intr.mat'.
#  Return both matrices as well.
def open3D_virtual_camera_window(photoW, photoH, params):
	geometry = read_point_cloud(params['cloud'])					#  Load point cloud into 'geometry'
	vis = Visualizer()

	#  Another quirk of Open3D is that too-large windows or windows that do not match the avowed width and
	#  height in the camera JSON file can cause it to crash.

	vis.create_window(window_name='Virtual Camera', width=photoW, height=photoH)
	if not os.path.exists('render.pointcloud.json'):				#  Make a point-cloud rendering parameters JSON
		build_rendering_JSON()
																	#  Now we're assured to have something to open
	vis.get_render_option().load_from_json('render.pointcloud.json')
	vis.add_geometry(geometry)										#  Add point cloud to visualizer
	controller = vis.get_view_control()								#  Get a controller

	build_camera_JSON(params, photoW, photoH)						#  Make "open3d.camera.json"
																	#  Not really a "trajectory," but... eh
	trajectory = read_pinhole_camera_trajectory('open3d.camera.json')
	controller.convert_from_pinhole_camera_parameters(trajectory.parameters[0])
	vis.run()

	#  Open3D has a KNOWN BUG that causes renderings of the RGB buffer to be larger and come from what looks like a
	#  different position. Until this is remedied, we stoop to the cheesy workaround of taking a screen-shot of the
	#  Visualizer window. Be sure that this is a bug because the depth rendering is correct!
	#  So line up a camera position you like, and just before you close the Visualizer window remember to take a
	#  screen-shot of just the window and name it 'rbg.png'. Obviously, don't make any changes to the camera
	#  position between taking the screen-shot and closing the window.

	'''																#  Be nice if this worked....
	image = vis.capture_screen_float_buffer(True)
	plt.imsave('rgb.png', np.asarray(image), dpi=1)
	'''

	print('\nDid you remember to take a picture and save it as "rgb.png"?\n')

																	#  The window closing is the script's signal
																	#  to render the depth map.
	depth = vis.capture_depth_float_buffer(False)					#  REMEMBER THAT THE DEPTH-MAP RENDERING IS THE
	plt.imsave('depth.png', np.asarray(depth), dpi=1)				#  CORRECT REFLECTION OF CAMERA PARAMETERS!!!
	np.save('depth.npy', np.asarray(depth))							#  Also save depth as FLOATS
	depth = np.asarray(depth)
																	#  Save camera extrinsics, intrinsics
	open3dParams = controller.convert_to_pinhole_camera_parameters()
	virtCamExt  = open3dParams.extrinsic
	virtCamIntr = open3dParams.intrinsic.intrinsic_matrix

	fstr  = str(virtCamIntr[0][0])+' '+str(virtCamIntr[0][1])+' '+str(virtCamIntr[0][2])+'\n'
	fstr += str(virtCamIntr[1][0])+' '+str(virtCamIntr[1][1])+' '+str(virtCamIntr[1][2])+'\n'
	fstr += str(virtCamIntr[2][0])+' '+str(virtCamIntr[2][1])+' '+str(virtCamIntr[2][2])
	fh = open('intr.mat', 'w')
	fh.write(fstr)
	fh.close()

	fstr  = str(virtCamExt[0][0])+' '+str(virtCamExt[0][1])+' '+str(virtCamExt[0][2])+' '+str(virtCamExt[0][3])+'\n'
	fstr += str(virtCamExt[1][0])+' '+str(virtCamExt[1][1])+' '+str(virtCamExt[1][2])+' '+str(virtCamExt[1][3])+'\n'
	fstr += str(virtCamExt[2][0])+' '+str(virtCamExt[2][1])+' '+str(virtCamExt[2][2])+' '+str(virtCamExt[2][3])+'\n'
	fstr += str(virtCamExt[3][0])+' '+str(virtCamExt[3][1])+' '+str(virtCamExt[3][2])+' '+str(virtCamExt[3][3])
	fh = open('ext.mat', 'w')
	fh.write(fstr)
	fh.close()

	vis.destroy_window()											#  Done with the virtual camera

	os.remove('open3d.camera.json')									#  Throw these away, too.
	os.remove('render.pointcloud.json')								#  They're temporary files.

	return virtCamIntr, virtCamExt

#  Return a list of 3D points
def clicker_RGB(intr, ext, params):
	if not os.path.exists('rgb.png'):								#  Did we create a file named rgb.png?
		print('Unable to find point cloud rendering "rgb.png"')
		return None

	if not os.path.exists('depth.npy'):								#  Did we create a file named depth.npy?
		print('Unable to find Z buffer "depth.npy"')
		return None
	else:
		depth = np.load('depth.npy')

	pts3D = []														#  Goal of this function is to fill this list
																	#  using interactive clicks and an extrinsic matrix

	args = ['./clicker', 'rgb.png' ]								#  Summon the clicker program
	output = subprocess.check_output(args)							#  Get its output
	imagePts = [x.split(',') for x in output.split()]				#  Parse four points from clicked photo
	for i in range(0, len(imagePts)):
		imagePts[i] = [float(x) for x in imagePts[i]]				#  Convert each comma-separated number to a float
	imagePts = imagePts[-4:]										#  Keep only the last four

	K_inv = inv(intr)												#  3x3
																	#  Undo the virtual camera
	for i in range(0, 4):
																	#  MATRIX lookup--not an IMAGE lookup!
																	#  So SWITCH INDICES!!
		z = depth.item(int(round(imagePts[i][1])), int(round(imagePts[i][0])))

		v = np.array( [[x] for x in imagePts[i]] + [[1.0]] )		#  Append [1.0] to make 3x1 column-vector
		v = K_inv.dot(v)											#  Multiply by inverse-K
		v *= z														#  Multiply by depth to get 3D point in CAMERA FRAME
		v = np.array([[v[0][0]], [v[1][0]], [v[2][0]], [1.0]])		#  Append [1.0] to make 4x1 column-vector
		v = inv(ext).dot(v)

		pts3D.append( (v[0][0], v[1][0], v[2][0]) )					#  Add 3D point to list

																	#  Great, now write this to file so we can be lazy
																	#  in the future.
	fstr = params['cloud'] + '\n'									#  First line is the name of the point cloud file
	for i in range(0, len(pts3D)):
		fstr += str(pts3D[i][0]) + ' ' + str(pts3D[i][1]) + ' ' + str(pts3D[i][2]) + '\n'
	fh = open('3D.txt', 'w')
	fh.write(fstr)
	fh.close()

	return pts3D

#  Return a list of 2D points
def clicker_photo(params):
	pts2D = []														#  Goal of this function is to fill this list
																	#  using interactive clicks

	args = ['./clicker', params['photo'] ]							#  Summon the clicker program
	output = subprocess.check_output(args)							#  Get its output
	imagePts = [x.split(',') for x in output.split()]				#  Parse four points from clicked photo
	for i in range(0, len(imagePts)):
		imagePts[i] = [float(x) for x in imagePts[i]]				#  Convert each comma-separated number to a float
	imagePts = imagePts[-4:]										#  Keep only the last four

	for i in range(0, 4):
		pts2D.append( (imagePts[i][0], imagePts[i][1]) )
																	#  Great, now write this to file so we can be lazy
																	#  in the future.
	fstr = params['photo'] + '\n'									#  First line is the name of the image file
	for i in range(0, len(pts2D)):
		fstr += str(pts2D[i][0]) + ' ' + str(pts2D[i][1]) + '\n'
	fh = open('2D.txt', 'w')
	fh.write(fstr)
	fh.close()

	return pts2D

#  Create a PLY file of the camera that took the given photo
def write_camera_pyramid(rotation, translation, params):
	if params['verbose']:
		print('Writing camera pyramid PLY file')
	plystr  = 'ply\n'
	plystr += 'format ascii 1.0\n'
	plystr += 'comment https://github.com/EricCJoyce\n'
	plystr += 'element vertex 5\n'
	plystr += 'property float x\n'
	plystr += 'property float y\n'
	plystr += 'property float z\n'
	plystr += 'element edge 8\n'
	plystr += 'property int vertex1\n'
	plystr += 'property int vertex2\n'
	plystr += 'property uchar red\n'
	plystr += 'property uchar green\n'
	plystr += 'property uchar blue\n'
	plystr += 'end_header\n'
	campoints = []
	campoints.append( np.array([0.0, 0.0, 0.0]) )					#  Push camera center
																	#  Push upper-left image plane corner
	campoints.append( np.array([ -params['Kmat'][0][2],  params['Kmat'][1][2], params['Kmat'][0][0] ]) )
																	#  Push upper-right image plane corner
	campoints.append( np.array([  params['Kmat'][0][2],  params['Kmat'][1][2], params['Kmat'][0][0] ]) )
																	#  Push lower-right image plane corner
	campoints.append( np.array([  params['Kmat'][0][2], -params['Kmat'][1][2], params['Kmat'][0][0] ]) )
																	#  Push lower-left image plane corner
	campoints.append( np.array([ -params['Kmat'][0][2], -params['Kmat'][1][2], params['Kmat'][0][0] ]) )

	R, _ = cv2.Rodrigues(rotation)
	t = -R.T.dot(translation)
	R = R.T

	for point in campoints:
		p = point / params['Kmat'][0][0]
		p = p * 0.1
		T = np.array([[R[0][0], R[0][1], R[0][2], t[0][0]], \
		              [R[1][0], R[1][1], R[1][2], t[1][0]], \
		              [R[2][0], R[2][1], R[2][2], t[2][0]], \
		              [0.0,     0.0,     0.0,     1.0]])
		p = T.dot( np.array([p[0], p[1], p[2], 1.0]))
		plystr += str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]) + '\n'

	plystr += '0 1 0 255 0\n'
	plystr += '0 2 0 255 0\n'
	plystr += '0 3 0 0 0\n'
	plystr += '0 4 0 0 0\n'
	plystr += '1 2 0 255 0\n'
	plystr += '2 3 0 0 0\n'
	plystr += '3 4 0 0 0\n'
	plystr += '4 1 0 0 0\n'

	fh = open('camerapose.ply', 'w')
	fh.write(plystr)
	fh.close()

	return

#  Build a projection matrix from the given rotation 3-vector 'r',
#  the given translation 3-vector 't', and the given intrinsic matrix 'K'
def buildProjectionMatrix(r, t, K):
	R, _ = cv2.Rodrigues(r)
	Ext = np.array([ [ R[0][0], R[0][1], R[0][2], t[0][0] ], \
	                 [ R[1][0], R[1][1], R[1][2], t[1][0] ], \
	                 [ R[2][0], R[2][1], R[2][2], t[2][0] ] ], dtype=np.float64)
	return K.dot(Ext)

#  Open3D requires JSON specs for everything.
#  Build a default rendering-settings file if one doesn't already exist.
def build_rendering_JSON():
	fh = open('render.pointcloud.json', 'w')
	fstr  = '{\n'
	fstr += '\t"background_color" : [ 1, 1, 1 ],\n'
	fstr += '\t"class_name" : "RenderOption",\n'
	fstr += '\t"default_mesh_color" : [ 0.69999999999999996, 0.69999999999999996, 0.69999999999999996 ],\n'
	fstr += '\t"image_max_depth" : 3000,\n'
	fstr += '\t"image_stretch_option" : 0,\n'
	fstr += '\t"interpolation_option" : 0,\n'
	fstr += '\t"light0_color" : [ 1, 1, 1 ],\n'
	fstr += '\t"light0_diffuse_power" : 0.66000000000000003,\n'
	fstr += '\t"light0_position" : [ 0, 0, 2 ],\n'
	fstr += '\t"light0_specular_power" : 0.20000000000000001,\n'
	fstr += '\t"light0_specular_shininess" : 100,\n'
	fstr += '\t"light1_color" : [ 1, 1, 1 ],\n'
	fstr += '\t"light1_diffuse_power" : 0.66000000000000003,\n'
	fstr += '\t"light1_position" : [ 0, 0, 2 ],\n'
	fstr += '\t"light1_specular_power" : 0.20000000000000001,\n'
	fstr += '\t"light1_specular_shininess" : 100,\n'
	fstr += '\t"light2_color" : [ 1, 1, 1 ],\n'
	fstr += '\t"light2_diffuse_power" : 0.66000000000000003,\n'
	fstr += '\t"light2_position" : [ 0, 0, -2 ],\n'
	fstr += '\t"light2_specular_power" : 0.20000000000000001,\n'
	fstr += '\t"light2_specular_shininess" : 100,\n'
	fstr += '\t"light3_color" : [ 1, 1, 1 ],\n'
	fstr += '\t"light3_diffuse_power" : 0.66000000000000003,\n'
	fstr += '\t"light3_position" : [ 0, 0, -2 ],\n'
	fstr += '\t"light3_specular_power" : 0.20000000000000001,\n'
	fstr += '\t"light3_specular_shininess" : 100,\n'
	fstr += '\t"light_ambient_color" : [ 0, 0, 0 ],\n'
	fstr += '\t"light_on" : true,\n'
	fstr += '\t"mesh_color_option" : 1,\n'
	fstr += '\t"mesh_shade_option" : 0,\n'
	fstr += '\t"mesh_show_back_face" : false,\n'
	fstr += '\t"mesh_show_wireframe" : false,\n'
	fstr += '\t"point_color_option" : 9,\n'
	fstr += '\t"point_show_normal" : false,\n'
	fstr += '\t"point_size" : 5,\n'
	fstr += '\t"show_coordinate_frame" : false,\n'
	fstr += '\t"version_major" : 1,\n'
	fstr += '\t"version_minor" : 0\n'
	fstr += '}'
	fh.write(fstr)
	fh.close()
	return

#  Open3D requires JSON specs for everything.
#  Turn the details in our nice, compact camera intrinsics file into a sprawl of tags, braces and brackets.
def build_camera_JSON(params, w, h):
	fh = open('open3d.camera.json', 'w')							#  This JSON stuff is a pain, and I don't want it
	fstr  = '{\n'													#  floating around our directory. Write numbers from
	fstr += '\t"class_name" : "PinholeCameraTrajectory",\n'			#  the K matrix to file, then erase that file when
	fstr += '\t"parameters" :\n'									#  we're done.
	fstr += '\t[\n'
	fstr += '\t\t{\n'
	fstr += '\t\t\t"class_name" : "PinholeCameraParameters",\n'
	fstr += '\t\t\t"extrinsic" :\n'
	fstr += '\t\t\t[\n'
	fstr += '\t\t\t\t'+str(1.0)+',\n'
	fstr += '\t\t\t\t'+str(0.0)+',\n'
	fstr += '\t\t\t\t'+str(0.0)+',\n'
	fstr += '\t\t\t\t'+str(0.0)+',\n'

	fstr += '\t\t\t\t'+str(0.0)+',\n'
	fstr += '\t\t\t\t'+str(1.0)+',\n'
	fstr += '\t\t\t\t'+str(0.0)+',\n'
	fstr += '\t\t\t\t'+str(0.0)+',\n'

	fstr += '\t\t\t\t'+str(0.0)+',\n'
	fstr += '\t\t\t\t'+str(0.0)+',\n'
	fstr += '\t\t\t\t'+str(1.0)+',\n'
	fstr += '\t\t\t\t'+str(0.0)+',\n'

	fstr += '\t\t\t\t'+str(0.0)+',\n'
	fstr += '\t\t\t\t'+str(0.0)+',\n'
	fstr += '\t\t\t\t'+str(0.0)+',\n'
	fstr += '\t\t\t\t'+str(1.0)+'\n'
	fstr += '\t\t\t],\n'
	fstr += '\t\t\t"intrinsic" :\n'
	fstr += '\t\t\t{\n'
	fstr += '\t\t\t\t"height" : '+str(h)+',\n'
	fstr += '\t\t\t\t"intrinsic_matrix" :\n'
	fstr += '\t\t\t\t[\n'
	fstr += '\t\t\t\t\t'+str(params['Kmat'][0][0])+',\n'
	fstr += '\t\t\t\t\t'+str(params['Kmat'][1][0])+',\n'
	fstr += '\t\t\t\t\t'+str(params['Kmat'][2][0])+',\n'

	fstr += '\t\t\t\t\t'+str(params['Kmat'][0][1])+',\n'
	fstr += '\t\t\t\t\t'+str(params['Kmat'][1][1])+',\n'
	fstr += '\t\t\t\t\t'+str(params['Kmat'][2][1])+',\n'

	fstr += '\t\t\t\t\t'+str(params['Kmat'][0][2])+',\n'
	fstr += '\t\t\t\t\t'+str(params['Kmat'][1][2])+',\n'
	fstr += '\t\t\t\t\t'+str(params['Kmat'][2][2])+'\n'
	fstr += '\t\t\t\t],\n'
	fstr += '\t\t\t\t"width" : '+str(w)+'\n'
	fstr += '\t\t\t},\n'
	fstr += '\t\t\t"version_major" : 1,\n'
	fstr += '\t\t\t"version_minor" : 0\n'
	fstr += '\t\t}\n'
	fstr += '\t],\n'
	fstr += '\t"version_major" : 1,\n'
	fstr += '\t"version_minor" : 0\n'
	fstr += '}'
	fh.write(fstr)
	fh.close()
	return

#  Read the given intrinsic matrix file and construct a K matrix accordingly.
#  The file we expect should have the following format:
#  Comments begin with a pound character, #
#  fx, separated by whitespace, followed by a real number, sets K[0][0]
#  fy, separated by whitespace, followed by a real number, sets K[1][1]
#  cx, separated by whitespace, followed by a real number, sets K[0][2]
#  cy, separated by whitespace, followed by a real number, sets K[1][2]
#  These can appear in any order, but values not set in this way will leave the initial values
#  in place: [[0, 0, 0],
#             [0, 0, 0],
#             [0, 0, 1]]
#  Also notice that the image has the final say. We allow that the target image may have been scaled
#  for memory constraints, so we take our lead from the calibration file, but amend that according
#  to the image we must work with.
def loadKFromFile(Kfilename, photo):
	fh = open(Kfilename, 'r')
	lines = fh.readlines()
	fh.close()
	w = None														#  These may not have been included
	h = None
	height, width, channels = photo.shape
	K = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
	for line in lines:
		arr = line.strip().split()
		if arr[0] != '#':											#  Ignore comments
			if arr[0] == 'fx':										#  Set fx
				K[0][0] = float(arr[1])
			elif arr[0] == 'fy':									#  Set fy
				K[1][1] = float(arr[1])
			elif arr[0] == 'cx':									#  Set cx
				K[0][2] = float(arr[1])
			elif arr[0] == 'cy':									#  Set cy
				K[1][2] = float(arr[1])
			elif arr[0] == 'w':										#  Save w
				w = float(arr[1])
			elif arr[0] == 'h':										#  Save h
				h = float(arr[1])

	a = width / w													#  Derive the scaling factor (arbitrarily) from width

	K[0][0] *= a													#  Scale fx
	K[1][1] *= a													#  Scale fy
	K[0][2] *= a													#  Scale cx
	K[1][2] *= a													#  Scale cy

	return K

#  Parse the command line and set variables accordingly
def parseRunParameters():
	Kmat = 'K.mat'													#  Default camera intrinsic matrix file
	showFeatures = False											#  Whether to a 3D reference file for features
	verbose = False
	output = []														#  All formats for output
	helpme = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-K', '-v', '-o', '-showFeatures', \
	         '-?', '-help', '--help']
	for i in range(3, len(sys.argv)):
		if sys.argv[i] in flags:
			argtarget = sys.argv[i]
			if argtarget == '-v':
				verbose = True
			elif argtarget == '-showFeatures':
				showFeatures = True
			elif argtarget == '-?' or argtarget == '-help' or argtarget == '--help':
				helpme = True
		else:
			argval = sys.argv[i]

			if argtarget is not None:
				if argtarget == '-K':								#  Following argument sets the intrinsicmatrix file
					Kmat = argval
				elif argtarget == '-o':								#  Following string is an output format
					if argval not in output:
						output.append(argval)

	params = {}
	params['Kmat'] = Kmat
	params['verbose'] = verbose
	params['output'] = output
	params['showFeatures'] = showFeatures
	params['helpme'] = helpme

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('Usage:  python point.py image-filename mesh-filename <options, each preceded by a flag>')
	print(' e.g.:  python point.py image.jpg mesh.obj -K SONY-DSLR-A580-P30mm.mat -v -iter 100000 -o P')
	print('Flags:  -K            following argument is the file containing camera intrinsic data.')
	print('                      The structure and necessary details in a camera file are outlined in README.md')
	print('        -v            enable verbosity')
	print('        -o            add an output format. Please see the Readme file for recognized formats.')
	print('        -showFeatures will generate a point cloud of the inferred 3D points.')
	print('        -?')
	print('        -help')
	print('        --help        displays this message.')
	return

if __name__ == '__main__':
	main()
