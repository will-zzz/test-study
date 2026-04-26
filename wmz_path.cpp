/*
 *  Copyright 2026 Will Zakielarz
 */

#include "include/GPath.h"
#include "include/GPathBuilder.h"
#include "include/GRect.h"
#include <algorithm>

GRect GPath::bounds() const {
    if (fPts.empty()) {
        return GRect::LTRB(0, 0, 0, 0);
    }
    // Find min/max
    float l = fPts[0].x, t = fPts[0].y, r = fPts[0].x, b = fPts[0].y;
    for (size_t i = 1; i < fPts.size(); i++) {
        l = std::min(l, fPts[i].x);
        t = std::min(t, fPts[i].y);
        r = std::max(r, fPts[i].x);
        b = std::max(b, fPts[i].y);
    }
    return GRect::LTRB(l, t, r, b);
}

void GPathBuilder::addRect(const GRect& r, GPathDirection dir) {
    // Start at top left
    moveTo(r.TL());
    if (dir == GPathDirection::kCW) {
        lineTo(r.TR());
        lineTo(r.BR());
        lineTo(r.BL());
    } else {
        lineTo(r.BL());
        lineTo(r.BR());
        lineTo(r.TR());
    }
}

void GPathBuilder::addPolygon(const GPoint pts[], int count) {
    if (count < 1) {
        return;
    }
    // moveTo first, lineTo the rest
    moveTo(pts[0]);
    for (int i = 1; i < count; i++) {
        lineTo(pts[i]);
    }
}

void GPathBuilder::addOval(const GRect& r, GPathDirection dir) {
    if (r.empty()) {
        return;
    }

    // Approximation constant
    float kappa = 0.55f;
    // Center of bounding rect
    float cx = r.cx();
    float cy = r.cy();
    float rx = r.width() * 0.5f;
    float ry = r.height() * 0.5f;
    // Offsets from the anchor points
    float ox = rx * kappa;
    float oy = ry * kappa;

    // Start at the right
    this->moveTo(cx + rx, cy);

    if (dir == GPathDirection::kCW) {
        // Clockwise
        this->cubicTo({cx + rx, cy + oy}, {cx + ox, cy + ry}, {cx, cy + ry});
        this->cubicTo({cx - ox, cy + ry}, {cx - rx, cy + oy}, {cx - rx, cy});
        this->cubicTo({cx - rx, cy - oy}, {cx - ox, cy - ry}, {cx, cy - ry});
        this->cubicTo({cx + ox, cy - ry}, {cx + rx, cy - oy}, {cx + rx, cy});
    } else {
        // Counter-clockwise
        this->cubicTo({cx + rx, cy - oy}, {cx + ox, cy - ry}, {cx, cy - ry});
        this->cubicTo({cx - ox, cy - ry}, {cx - rx, cy - oy}, {cx - rx, cy});
        this->cubicTo({cx - rx, cy + oy}, {cx - ox, cy + ry}, {cx, cy + ry});
        this->cubicTo({cx + ox, cy + ry}, {cx + rx, cy + oy}, {cx + rx, cy});
    }
}
