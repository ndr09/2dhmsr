/*
 * Copyright (C) 2019 Eric Medvet <eric.medvet@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package it.units.erallab.hmsrobots.controllers;

import it.units.erallab.hmsrobots.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import java.util.EnumSet;
import java.util.List;

/**
 *
 * @author Eric Medvet <eric.medvet@gmail.com>
 */
public abstract class ClosedLoopController implements Controller {

  private final Grid<List<Voxel.Sensor>> sensorsGrid;

  public ClosedLoopController(Grid<List<Voxel.Sensor>> sensorsGrid) {
    this.sensorsGrid = sensorsGrid;
  }

  public Grid<List<Voxel.Sensor>> getSensorsGrid() {
    return sensorsGrid;
  }

  protected double[] collectInputs(Voxel voxel, List<Voxel.Sensor> sensors) {
    double[] values = new double[sensors.size()];
    collectInputs(voxel, sensors, values, 0);
    return values;
  }
  
  protected void collectInputs(Voxel voxel, List<Voxel.Sensor> sensors, double[] values, int c) {
    for (Voxel.Sensor sensor : sensors) {
      values[c] = voxel.getSensorReading(sensor);
      c = c + 1;
    }
  }

}