/*
 *  Copyright 2026 Will Zakielarz
 */

#include "include/GMatrix.h"
#include <cmath>

GMatrix::GMatrix() {
    // Identity: [1 0 0; 0 1 0]
    fMat[0] = 1; fMat[1] = 0; fMat[2] = 0;
    fMat[3] = 1; fMat[4] = 0; fMat[5] = 0;
}

GMatrix GMatrix::Translate(float tx, float ty) {
    return GMatrix(1, 0, tx, 0, 1, ty);
}

GMatrix GMatrix::Scale(float sx, float sy) {
    return GMatrix(sx, 0, 0, 0, sy, 0);
}

GMatrix GMatrix::Rotate(float radians) {
    // Float functions
    float c = cosf(radians);
    float s = sinf(radians);
    
    return GMatrix(c, -s, 0, s, c, 0);
}

GMatrix GMatrix::Concat(const GMatrix& a, const GMatrix& b) {
    // Matrix multiplication
    return GMatrix(
        a[0] * b[0] + a[2] * b[1],  a[0] * b[2] + a[2] * b[3],  a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[0] + a[3] * b[1],  a[1] * b[2] + a[3] * b[3],  a[1] * b[4] + a[3] * b[5] + a[5]
    );
}

std::optional<GMatrix> GMatrix::invert() const {
    float det = fMat[0] * fMat[3] - fMat[1] * fMat[2];
    if (det == 0) {
        return {};
    }

    float inv = 1.0f / det;
    
    return GMatrix(
        fMat[3] * inv,  -fMat[2] * inv,  (fMat[2] * fMat[5] - fMat[3] * fMat[4]) * inv,
        -fMat[1] * inv,  fMat[0] * inv,   (fMat[1] * fMat[4] - fMat[0] * fMat[5]) * inv
    );
}

void GMatrix::mapPoints(GPoint dst[], const GPoint src[], int count) const {
    for (int i = 0; i < count; i++) {
        float x = src[i].x, y = src[i].y;

        dst[i].x = fMat[0] * x + fMat[2] * y + fMat[4];
        dst[i].y = fMat[1] * x + fMat[3] * y + fMat[5];
    }
}
