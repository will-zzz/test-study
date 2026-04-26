/*
 *  Copyright 2025 Will Zakielarz
 */

#ifndef _g_starter_canvas_h_
#define _g_starter_canvas_h_

#include "include/GCanvas.h"
#include "include/GPath.h"
#include "include/GPoint.h"
#include "include/GColor.h"
#include "include/GPaint.h"
#include "include/GBitmap.h"
#include "include/GMatrix.h"
#include <vector>

class WMZCanvas : public GCanvas {
public:
    WMZCanvas(const GBitmap& device) : fDevice(device), fCTM() {}

    void clear(const GColor&) override;
    void drawLine(GPoint, GPoint, const GPaint&) override;
    void drawRect(const GRect&, const GPaint&) override;
    void drawConvexPolygon(const GPoint[], int count, const GPaint&) override;
    void drawPath(const GPath&, const GPaint&) override;
    void drawMesh(const GPoint verts[], const GColor colors[], const GPoint texs[], int count, const int indices[], const GPaint&) override;
    void drawQuad(const GPoint verts[4], const GColor colors[4], const GPoint texs[4], int level, const GPaint&) override;
    void save() override;
    void restore() override;
    void concat(const GMatrix&) override;

private:
    // Note: we store a copy of the bitmap
    const GBitmap fDevice;
    // Current transformation matrix
    GMatrix fCTM;
    // Transformation matrix stack
    std::vector<GMatrix> fCTMStack;

    void place_pixel(int x, int y, GPixel src, GBlendMode mode);
};

#endif