from PIL import Image
import random
import numpy
import requests
import shutil
import urllib
import math
import sys

#Cluster and Kmeans are borrowed from ZeevG's implementation here: https://github.com/ZeevG/python-dominant-image-colour/blob/master/kmeans.py
class Cluster(object):

    def __init__(self):
        self.pixels = []
        self.centroid = None

    def addPoint(self, pixel):
        self.pixels.append(pixel)

    def setNewCentroid(self):

        R = [colour[0] for colour in self.pixels]
        G = [colour[1] for colour in self.pixels]
        B = [colour[2] for colour in self.pixels]

        if(len(R) != 0):
            R = sum(R) / len(R)
        else:
            R = 0
        if(len(G) != 0):
            G = sum(G) / len(G)
        else:
            G = 0
        if(len(B) != 0):
            B = sum(B) / len(B)
        else:
            B = 0
        self.centroid = (R, G, B)
        self.pixels = []

        return self.centroid


class Kmeans(object):

    def __init__(self, k=3, colorSet=[], size=(300, 300), max_iterations=5, min_distance=5.0):
        self.k = k
        self.colorSet = colorSet;
        self.max_iterations = max_iterations
        self.min_distance = min_distance
        self.size = size

    def run(self, image):
        self.image = image
        self.image.thumbnail(self.size)
        self.pixels = numpy.array(image.getdata(), dtype=numpy.uint8)

        self.clusters = [None for i in range(self.k)]
        self.oldClusters = None

        randomPixels = random.sample(self.pixels, self.k)

        for idx in range(self.k):
            self.clusters[idx] = Cluster()
            self.clusters[idx].centroid = randomPixels[idx]

        iterations = 0

        while self.shouldExit(iterations) is False:

            self.oldClusters = [cluster.centroid for cluster in self.clusters]

            print iterations

            for pixel in self.pixels:
                self.assignClusters(pixel)

            self.clusters = sorted(self.clusters, key = lambda x: len(x.pixels))

            for cluster in self.clusters:
                cluster.setNewCentroid()

            iterations += 1

        return [cluster.centroid for cluster in self.clusters]

    def assignClusters(self, pixel):
        shortest = float('Inf')
        for cluster in self.clusters:
            distance = self.calcDistance(cluster.centroid, pixel)
            if distance < shortest:
                shortest = distance
                nearest = cluster

        nearest.addPoint(pixel)

    def calcDistance(self, a, b):

        result = numpy.sqrt(sum((a - b) ** 2))
        return result

    def shouldExit(self, iterations):

        if self.oldClusters is None:
            return False

        for idx in range(self.k):
            dist = self.calcDistance(
                numpy.array(self.clusters[idx].centroid),
                numpy.array(self.oldClusters[idx])
            )
            if dist < self.min_distance:
                return True

        if iterations <= self.max_iterations:
            return False

        return True

    def showClustering(self, fileName):
        clusterMap = {}
        index = 0
        for i in self.clusters:
            clusterMap[index] = i
            index += 1

        colors = self.colorSet
        localPixels = [None] * len(self.image.getdata())

        for idx, pixel in enumerate(self.pixels):
                shortest = float('Inf')
                for cindex in clusterMap:
                    cluster = clusterMap[cindex]
                    distance = self.calcDistance(cluster.centroid, pixel)
                    if distance < shortest:
                        shortest = distance
                        nearest = cindex
                        wheel = cluster.centroid

                diffrnz = dev(wheel, pixel)
                newk = div(diffrnz, colors[nearest])
                localPixels[idx] = newk

        w, h = self.image.size
        localPixels = numpy.asarray(localPixels)\
            .astype('uint8')\
            .reshape((h, w, 4))

        colourMap = Image.fromarray(localPixels, "RGBA")
        out = Image.blend(self.image.convert("RGBA"), colourMap, .9)
        out.save("colored" + fileName + ".png", "PNG")
        print "Done"

def dev(a, b):
    out1 = 0
    out2 = 0
    out3 = 0
    for i in xrange(len(a)):
        if(b[i] >= a[i]):
            out1 += (b[i] - a[i]) ** 2
        else:
            out2 += (b[i] - a[i]) ** 2

    if out1 > out2:
        out3 = 1
    else:
        out3 = -1

    out = math.sqrt(out1 + out2)
        
    return [out, out3]

def div(a, b):
    b += (int(128 + a[0]*.5*a[1]),)
    return b

def getPicture1(location, size):
    sizeArr = [size, size]

    outName = str(location).replace(" ", "_")

    mapsApiKey = "AIzaSyB3JXEv4fobNoCYCwCyKYNsgK9EakXK4fs"
    baseUrl = "https://maps.googleapis.com/maps/api/streetview?"

    queries = {}
    queries["key"] = mapsApiKey
    queries["location"] = location
    queries["size"] = str(sizeArr[0]) + "x" + str(sizeArr[1])

    queries = urllib.urlencode(queries)

    queryUrl = baseUrl + queries

    response = requests.get(queryUrl, stream=True)
    with open(outName + '.jpg', 'wn') as out_file:
       shutil.copyfileobj(response.raw, out_file)
    del response

def getPicture(location, size, extraQueries):
    sizeArr = [size, size]

    outName = str(location).replace(" ", "_")

    mapsApiKey = "AIzaSyB3JXEv4fobNoCYCwCyKYNsgK9EakXK4fs"
    baseUrl = "https://maps.googleapis.com/maps/api/streetview?"

    queries = {}

    for j in extraQueries:
        queries[j] = extraQueries[j]        
    queries["key"] = mapsApiKey
    queries["location"] = location
    queries["size"] = str(sizeArr[0]) + "x" + str(sizeArr[1])

    queries = urllib.urlencode(queries)

    queryUrl = baseUrl + queries

    response = requests.get(queryUrl, stream=True)
    with open(outName + '.jpg', 'wn') as out_file:
       shutil.copyfileobj(response.raw, out_file)
    del response

def loadColors():
    colors = {}
    arc = []
    
    with open('colorschemes') as f:
        ar = []
        
        for line in f:
            if line.strip() != "":
                ar.append(line.strip())
            else:
                arc.append(ar)
                ar = []
                
    arc.append(ar)
    
    for j in arc:
        c = []
        
        for k in j[1:]:
            c.append(tuple(k.split(",")))
        colors[j[0]] = c
    
    return colors

def main1(location, colors, maxiter):
    colorSets = loadColors()
    
    if colorSets[colors] != None:
        colorSet = colorSets[colors]
    
    kt = len(colorSet)
    
    image = Image.open(location)

    k = Kmeans(kt, colorSet, image.size, maxiter)

    result = k.run(image)

    k.showClustering(colors + "_" + location.split(".")[0])

if __name__ == "__main__":
    main1(sys.argv[1], sys.argv[2], int(sys.argv[3]))
