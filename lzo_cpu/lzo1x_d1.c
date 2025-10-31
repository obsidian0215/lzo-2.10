/* lzo1x_d1.c -- LZO1X decompression (local copy for lzo_cpu)
 *
 * This file is derived from the upstream LZO distribution and kept here so
 * that the CPU-only tool can be built without reaching into the toplevel
 * src/ directory. The original copyright and licensing terms are preserved
 * below.
 */

/* lzo1x_d1.c -- LZO1X decompression

   This file is part of the LZO real-time data compression library.

   Copyright (C) 1996-2017 Markus Franz Xaver Johannes Oberhumer
   All Rights Reserved.

   The LZO library is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation; either version 2 of
   the License, or (at your option) any later version.

   The LZO library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with the LZO library; see the file COPYING.
   If not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.

   Markus F.X.J. Oberhumer
   <markus@oberhumer.com>
   http://www.oberhumer.com/opensource/lzo/
 */


#include "config1x.h"

#undef LZO_TEST_OVERRUN
#define DO_DECOMPRESS       lzo1x_decompress

/* Use the local copy of decompression chunk from lzo_cpu/src to stay
 * completely self-contained.
 */
#include "src/lzo1x_d.ch"

/* vim:set ts=4 sw=4 et: */
