# Copyright 2017 Hieu Tran. All Rights Reserved.
#
# DeepNovo is publicly available for non-commercial uses.
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
cimport numpy as np
cimport cython

import deepnovo_config

mass_ID_np = deepnovo_config.mass_ID_np
cdef int GO_ID = deepnovo_config.GO_ID
cdef int EOS_ID = deepnovo_config.EOS_ID
cdef float mass_H2O = deepnovo_config.mass_H2O
cdef float mass_NH3 = deepnovo_config.mass_NH3
cdef float mass_H = deepnovo_config.mass_H
cdef int SPECTRUM_RESOLUTION = deepnovo_config.SPECTRUM_RESOLUTION
cdef int WINDOW_SIZE = deepnovo_config.WINDOW_SIZE
cdef int vocab_size = deepnovo_config.vocab_size
cdef int num_ion = deepnovo_config.num_ion
cdef int MZ_SIZE = deepnovo_config.MZ_SIZE



@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False) # turn off negative index wrapping
def get_ion_index(peptide_mass, prefix_mass, direction):
  """

  :param peptide_mass: neutral mass of a peptide
  :param prefix_mass:
  :param direction: 0 for forward, 1 for backward
  :return: an int32 ndarray of shape [26, 8], each element represent a index of the spectrum embbeding matrix. for out
  of bound position, the index is 0
  """
  if direction == 0:
    candidate_b_mass = prefix_mass + mass_ID_np
    candidate_y_mass = peptide_mass - candidate_b_mass
  elif direction == 1:
    candidate_y_mass = prefix_mass + mass_ID_np
    candidate_b_mass = peptide_mass - candidate_y_mass

  # b-ions
  candidate_b_H2O = candidate_b_mass - mass_H2O
  candidate_b_NH3 = candidate_b_mass - mass_NH3
  candidate_b_plus2_charge1 = ((candidate_b_mass + 2 * mass_H) / 2
                               - mass_H)

  # y-ions
  candidate_y_H2O = candidate_y_mass - mass_H2O
  candidate_y_NH3 = candidate_y_mass - mass_NH3
  candidate_y_plus2_charge1 = ((candidate_y_mass + 2 * mass_H) / 2
                               - mass_H)

  # ion_2
  #~   b_ions = [candidate_b_mass]
  #~   y_ions = [candidate_y_mass]
  #~   ion_mass_list = b_ions + y_ions

  # ion_8
  b_ions = [candidate_b_mass,
            candidate_b_H2O,
            candidate_b_NH3,
            candidate_b_plus2_charge1]
  y_ions = [candidate_y_mass,
            candidate_y_H2O,
            candidate_y_NH3,
            candidate_y_plus2_charge1]
  ion_mass_list = b_ions + y_ions
  ion_mass = np.array(ion_mass_list, dtype=np.float32)  # 8 by 26

  # ion locations
  ion_location = np.ceil(ion_mass * SPECTRUM_RESOLUTION).astype(np.int64) # 8 by 26

  in_bound_mask = np.logical_and(
      ion_location > 0,
      ion_location <= MZ_SIZE).astype(np.int64)
  ion_location = ion_location * in_bound_mask  # 8 by 26, out of bound index would have value 0
  return ion_location.transpose()  # 26 by 8

@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False) # turn off negative index wrapping
cdef void copy_values(float[:,:,:] candidate_intensity_view, float[:] spectrum_view, int[:,:] location_sub, int i1, int i2):
  cdef int j
  for j in range(WINDOW_SIZE):
    candidate_intensity_view[i2, i1, j] = spectrum_view[location_sub[i1, i2] + j]


@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False) # turn off negative index wrapping
def get_location(peptide_mass, prefix_mass, direction):
  """

  :param peptide_mass: neutral mass of a peptide
  :param prefix_mass:
  :param direction: 0 for forward, 1 for backward
  :return:
  """
  if direction == 0:
    candidate_b_mass = prefix_mass + mass_ID_np
    candidate_y_mass = peptide_mass - candidate_b_mass
  elif direction == 1:
    candidate_y_mass = prefix_mass + mass_ID_np
    candidate_b_mass = peptide_mass - candidate_y_mass
  
  # b-ions
  candidate_b_H2O = candidate_b_mass - mass_H2O
  candidate_b_NH3 = candidate_b_mass - mass_NH3
  candidate_b_plus2_charge1 = ((candidate_b_mass + 2 * mass_H) / 2
                               - mass_H)

  # y-ions
  candidate_y_H2O = candidate_y_mass - mass_H2O
  candidate_y_NH3 = candidate_y_mass - mass_NH3
  candidate_y_plus2_charge1 = ((candidate_y_mass + 2 * mass_H) / 2
                               - mass_H)

  # ion_2
  #~   b_ions = [candidate_b_mass]
  #~   y_ions = [candidate_y_mass]
  #~   ion_mass_list = b_ions + y_ions

  # ion_8
  b_ions = [candidate_b_mass,
            candidate_b_H2O,
            candidate_b_NH3,
            candidate_b_plus2_charge1]
  y_ions = [candidate_y_mass,
            candidate_y_H2O,
            candidate_y_NH3,
            candidate_y_plus2_charge1]
  ion_mass_list = b_ions + y_ions
  ion_mass = np.array(ion_mass_list, dtype=np.float32)  # 8 by 26

  # ion locations
  location_sub50 = np.rint(ion_mass * SPECTRUM_RESOLUTION).astype(np.int32) # TODO(nh2tran): line-too-long
  # location_sub50 = np.int32(ion_mass * SPECTRUM_RESOLUTION)
  location_sub50 -= (WINDOW_SIZE // 2)
  location_plus50 = location_sub50 + WINDOW_SIZE
  ion_id_rows, aa_id_cols = np.nonzero(np.logical_and(
      location_sub50 >= 0,
      location_plus50 <= MZ_SIZE))
  return ion_id_rows, aa_id_cols, location_sub50, location_plus50

@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False) # turn off negative index wrapping
def get_candidate_intensity(float[:] spectrum_original, peptide_mass, prefix_mass, direction):
  """TODO(nh2tran): docstring."""
  ion_id_rows, aa_id_cols, location_sub50, location_plus50 = get_location(peptide_mass, prefix_mass, direction)
  # candidate_intensity
  candidate_intensity = np.zeros(shape=(vocab_size,
                                        num_ion,
                                        WINDOW_SIZE),
                                 dtype=np.float32)
  cdef int [:,:] location_sub50_view = location_sub50
  cdef int [:,:] location_plus50_view = location_plus50
  cdef float [:,:,:] candidate_intensity_view = candidate_intensity
  cdef int[:] row = ion_id_rows.astype(np.int32)
  cdef int[:] col = aa_id_cols.astype(np.int32)
  cdef int index
  for index in range(ion_id_rows.size):
    if col[index] < 3:
      continue
    copy_values(candidate_intensity_view, spectrum_original, location_sub50_view, row[index], col[index])
  # PAD/GO/EOS
  # candidate_intensity[deepnovo_config.PAD_ID].fill(0.0)
  # candidate_intensity[FIRST_LABEL].fill(0.0)
  # candidate_intensity[LAST_LABEL].fill(0.0)
  #~ b_ion_count = len(b_ions)
  #~ if (direction==0):
    #~ candidate_intensity[LAST_LABEL,b_ion_count:].fill(0.0)
  #~ elif (direction==1):
    #~ candidate_intensity[LAST_LABEL,:b_ion_count].fill(0.0)

  #~ for aa_id in ([LAST_LABEL] + range(3,deepnovo_config.vocab_size)):
    #~ for ion_id in range(deepnovo_config.num_ion):
      #~ location_sub50 = location_sub50_list[ion_id][aa_id]
      #~ #
      #~ if (location_sub50 > 0):
        #~ candidate_intensity[aa_id,ion_id] = spectrum_original[location_sub50:location_sub50+deepnovo_config.WINDOW_SIZE]

  # Nomalization to [0, 1]
  max_intensity = np.max(candidate_intensity)
  if max_intensity > 1.0:
    candidate_intensity /= max_intensity
  # Nomalization to N(0,1): tf.image.per_image_whitening
#~   adjusted_stddev = max(np.std(candidate_intensity), 1.0/math.sqrt(candidate_intensity.size))
#~   candidate_intensity = (candidate_intensity-np.mean(candidate_intensity)) / adjusted_stddev
  return candidate_intensity


def process_spectrum(spectrum_mz_list, spectrum_intensity_list, peptide_mass):
  """TODO(nh2tran): docstring."""

  # neutral mass, location, assuming ion charge z=1
  charge = 1.0
  spectrum_mz = np.array(spectrum_mz_list, dtype=np.float32)
  neutral_mass = spectrum_mz - charge*deepnovo_config.mass_H
  neutral_mass_location = np.rint(neutral_mass * deepnovo_config.SPECTRUM_RESOLUTION).astype(np.int32)
  cdef int [:] neutral_mass_location_view = neutral_mass_location

  # intensity
  spectrum_intensity = np.array(spectrum_intensity_list, dtype=np.float32)
  # log-transform
#~   spectrum_intensity = np.log(spectrum_intensity)
  # find max intensity value for normalization and to assign to special locations
  spectrum_intensity_max = np.max(spectrum_intensity)
  # no normalization for each individual spectrum, we'll do it for multi-spectra
#~   norm_intensity = spectrum_intensity / spectrum_intensity_max
  norm_intensity = spectrum_intensity / (spectrum_intensity_max + 1e-6)
  cdef float [:] norm_intensity_view = norm_intensity

  # fill spectrum holders
  spectrum_holder = np.zeros(shape=deepnovo_config.MZ_SIZE, dtype=np.float32)
  cdef float [:] spectrum_holder_view = spectrum_holder
  # note that different peaks may fall into the same location, hence loop +=
  cdef int index
  for index in range(neutral_mass_location.size):
#~     spectrum_holder_view[neutral_mass_location_view[index]] += norm_intensity_view[index]
    spectrum_holder_view[neutral_mass_location_view[index]] = max(spectrum_holder_view[neutral_mass_location_view[index]],
                                                                     norm_intensity_view[index])
  spectrum_original_forward = np.copy(spectrum_holder)
  spectrum_original_backward = np.copy(spectrum_holder)

  # add complement
  complement_mass = peptide_mass - neutral_mass
  complement_mass_location = np.rint(complement_mass * deepnovo_config.SPECTRUM_RESOLUTION).astype(np.int32) # TODO(nh2tran): line-too-long
  cdef int [:] complement_mass_location_view = complement_mass_location
#~   cdef int index
  for index in np.nonzero(complement_mass_location > 0)[0]:
    spectrum_holder_view[complement_mass_location_view[index]] += norm_intensity_view[index]

  # peptide_mass
  spectrum_original_forward[int(round(peptide_mass * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0
  spectrum_original_backward[int(round(peptide_mass * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0

  # N-terminal, b-ion, peptide_mass_C
  # append N-terminal
  mass_N = deepnovo_config.mass_N_terminus - deepnovo_config.mass_H
  spectrum_holder[int(round(mass_N * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0
  # append peptide_mass_C
  mass_C = deepnovo_config.mass_C_terminus + deepnovo_config.mass_H
  peptide_mass_C = peptide_mass - mass_C
  spectrum_holder[int(round(peptide_mass_C * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0
  spectrum_original_forward[int(round(peptide_mass_C * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0

  # C-terminal, y-ion, peptide_mass_N
  # append C-terminal
  mass_C = deepnovo_config.mass_C_terminus + deepnovo_config.mass_H
  spectrum_holder[int(round(mass_C * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0
  # append peptide_mass_N
  mass_N = deepnovo_config.mass_N_terminus - deepnovo_config.mass_H
  peptide_mass_N = peptide_mass - mass_N
  spectrum_holder[int(round(peptide_mass_N * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0
  spectrum_original_backward[int(round(peptide_mass_N * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0

  return spectrum_holder, spectrum_original_forward, spectrum_original_backward


def process_peaks(spectrum_mz_list, spectrum_intensity_list, peptide_mass):
  """

  :param spectrum_mz_list:
  :param spectrum_intensity_list:
  :param peptide_mass: peptide neutral mass
  :return:
    peak_location: int64, [N]
    peak_intensity: float32, [N]
  """

  charge = 1.0
  spectrum_mz = np.array(spectrum_mz_list, dtype=np.float32)
  neutral_mass = spectrum_mz - charge*deepnovo_config.mass_H
  neutral_mass_location = np.ceil(neutral_mass * deepnovo_config.SPECTRUM_RESOLUTION).astype(np.int64)
  cdef long [:] neutral_mass_location_view = neutral_mass_location

  # intensity
  spectrum_intensity = np.array(spectrum_intensity_list, dtype=np.float32)
  # log-transform
#~   spectrum_intensity = np.log(spectrum_intensity)
  # find max intensity value for normalization and to assign to special locations
  spectrum_intensity_max = np.max(spectrum_intensity)
  # no normalization for each individual spectrum, we'll do it for multi-spectra
  norm_intensity = spectrum_intensity / spectrum_intensity_max
  cdef float [:] norm_intensity_view = norm_intensity

  # fill spectrum holders
  spectrum_holder = np.zeros(shape=deepnovo_config.MZ_SIZE, dtype=np.float32)
  cdef float [:] spectrum_holder_view = spectrum_holder
  # note that different peaks may fall into the same location, hence loop +=
  cdef int index
  for index in range(neutral_mass_location.size):
    spectrum_holder_view[neutral_mass_location_view[index]] = max(spectrum_holder_view[neutral_mass_location_view[index]], # TODO(nh2tran): line-too-long
                                                                     norm_intensity_view[index])

  # peptide_mass
  spectrum_holder_view[int(np.ceil(peptide_mass * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0

  # N-terminal, b-ion, peptide_mass_C
  # append N-terminal
  mass_N = deepnovo_config.mass_N_terminus - deepnovo_config.mass_H
  spectrum_holder_view[int(np.ceil(mass_N * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0
  # append peptide_mass_C
  mass_C = deepnovo_config.mass_C_terminus + deepnovo_config.mass_H
  peptide_mass_C = peptide_mass - mass_C
  spectrum_holder_view[int(np.ceil(peptide_mass_C * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0

  # C-terminal, y-ion, peptide_mass_N
  # append C-terminal
  mass_C = deepnovo_config.mass_C_terminus + deepnovo_config.mass_H
  spectrum_holder_view[int(np.ceil(mass_C * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0
  # append peptide_mass_N
  mass_N = deepnovo_config.mass_N_terminus - deepnovo_config.mass_H
  peptide_mass_N = peptide_mass - mass_N
  spectrum_holder_view[int(np.ceil(peptide_mass_N * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0
  spectrum_holder_view[0] = 0.0

  top_N_indices = np.argpartition(spectrum_holder, -deepnovo_config.MAX_NUM_PEAK)[-deepnovo_config.MAX_NUM_PEAK:]
  intensity = spectrum_holder[top_N_indices]
  pos_intensity_mask = intensity > 1e-6
  top_N_indices[~pos_intensity_mask] = 0
  intensity[~pos_intensity_mask] = 0.
  return top_N_indices, intensity
