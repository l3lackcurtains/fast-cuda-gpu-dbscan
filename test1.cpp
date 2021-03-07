#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <vector>
using namespace std;

#define THREAD_BLOCKS 6

int main() {
  int collisionMatrix[THREAD_BLOCKS][THREAD_BLOCKS] = {
      {1, 0, 0, 1, 0, 0},
      {0, 1, 1, 0, 0, 0},
      {0, 0, 1, 1, 0, 1},
      {0, 0, 0, 1, 1, 0},
  };
  int map[THREAD_BLOCKS][THREAD_BLOCKS];
  for (int x = 0; x < THREAD_BLOCKS; x++) {
    for (int y = 0; y < THREAD_BLOCKS; y++) {
      map[x][y] = -1;
    }
  }
  for (int x = THREAD_BLOCKS - 1; x >= 0; x--) {
    int count = 0;
    for (int y = THREAD_BLOCKS - 1; y >= 0; y--) {
      if (collisionMatrix[x][y] == 1 && x != y) {
        cout << x << " - " << y << endl;
        map[x][count++] = y;
      }
    }
  }

  for (int x = THREAD_BLOCKS - 1; x >= 0; x--) {
    int smallest = THREAD_BLOCKS;
    for (int y = 0; y < THREAD_BLOCKS; y++) {
      if (map[x][y] == -1) break;
      if (map[x][y] < smallest) smallest = map[x][y];
    }

    cout << endl;
  }

  for (int x = 0; x < THREAD_BLOCKS; x++) {
    cout << x << ": ";
    for (int y = 0; y < THREAD_BLOCKS; y++) {
      cout << map[x][y] << " ";
    }
    cout << endl;
  }

  cout << "############################" << endl;

  std::map<int, int> colMap;
  std::set<int> blockSet;
  for (int i = 0; i < THREAD_BLOCKS; i++) {
    colMap[i] = i;
    blockSet.insert(i);
  }
  std::set<int>::iterator it;
  do {
    it = blockSet.begin();
    int curBlock = *it;
    std::set<int> expansionQueue;
    std::set<int> finalQueue;
    finalQueue.insert(curBlock);
    expansionQueue.insert(curBlock);
    do {
      it = expansionQueue.begin();
      int expandBlock = *it;
      expansionQueue.erase(it);
      blockSet.erase(expandBlock);
      for (int x = 0; x < THREAD_BLOCKS; x++) {
        if (x == expandBlock) continue;
        if ((collisionMatrix[expandBlock][x] == 1 ||
             collisionMatrix[x][expandBlock]) &&
            blockSet.find(x) != blockSet.end()) {
          expansionQueue.insert(x);
          finalQueue.insert(x);
        }
      }
    } while (expansionQueue.empty() == 0);

    for (it = finalQueue.begin(); it != finalQueue.end(); ++it) {
      colMap[*it] = curBlock;
    }
  } while (blockSet.empty() == 0);

  for (int i = 0; i < THREAD_BLOCKS; i++) {
    cout << i << ": " << colMap[i] << endl;
  }

  return 0;
}