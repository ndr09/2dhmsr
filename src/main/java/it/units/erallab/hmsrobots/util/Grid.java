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
package it.units.erallab.hmsrobots.util;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Eric Medvet <eric.medvet@gmail.com>
 */
public class Grid<T> {

  private final List<T> ts;
  private final int w;
  private final int h;

  public Grid(int w, int h, T[] t) {
    this.w = w;
    this.h = h;
    ts = new ArrayList<>(w * h);
    for (int i = 0; i < w * h; i++) {
      if ((t != null) && (i < t.length)) {
        ts.add(t[i]);
      } else {
        ts.add(null);
      }
    }
  }

  public T get(int x, int y) {
    if ((x < 0) || (x >= w)) {
      return null;
    }
    if ((y < 0) || (y >= h)) {
      return null;
    }
    return ts.get((y * w) + x);
  }

  public void set(int x, int y, T t) {
    ts.set((y * w) + x, t);
  }

  public int getW() {
    return w;
  }

  public int getH() {
    return h;
  }

  public static <K> Grid<K> create(int w, int h, K k) {
    Grid<K> grid = new Grid<>(w, h, null);
    for (int x = 0; x < grid.getW(); x++) {
      for (int y = 0; y < grid.getH(); y++) {
        grid.set(x, y, k);
      }
    }
    return grid;
  }

  public static <K> Grid<K> create(int w, int h) {
    return create(w, h, (K) null);
  }

  public static <K> Grid<K> create(Grid<?> other) {
    return create(other.getW(), other.getH());
  }

  public static <K> Grid<K> copy(Grid<K> other) {
    Grid<K> grid = Grid.create(other);
    for (int x = 0; x < grid.w; x++) {
      for (int y = 0; y < grid.h; y++) {
        grid.set(x, y, other.get(x, y));
      }
    }
    return grid;
  }

}
