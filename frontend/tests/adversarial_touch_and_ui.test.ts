import { describe, expect, it } from "vitest";

describe("Furniture Suggestion Curation & Data Contracts", () => {
  const getSuggestions = (objectName: string) => {
    const name = objectName.toLowerCase();
    // 1. Small tabletop & decor items (highest specificity)
    if (name.includes("mug") || name.includes("cup") || name.includes("vase") || name.includes("pot") || name.includes("bowl")) {
      return [
        { icon: "🏺", label: "Terracotta Vase", prompt: "Organic handcrafted raw terracotta ceramic vase" },
        { icon: "✨", label: "Amber Fluted Glass", prompt: "Hand-blown fluted amber glass with golden reflections" },
        { icon: "☕", label: "Sage Ceramic Mug", prompt: "Artisanal speckled matte sage green stoneware mug" },
        { icon: "🖤", label: "Obsidian Stoneware", prompt: "Architectural matte black volcanic stone vessel" },
      ];
    }
    // 2. Wall-mounted Art & Canvas (specific over raw walls)
    if (name.includes("canvas") || name.includes("art") || name.includes("frame") || name.includes("painting")) {
      return [
        { icon: "🎨", label: "Abstract Oil", prompt: "Large vibrant abstract expressionist canvas with textured impasto" },
        { icon: "🌻", label: "Golden Sunflowers", prompt: "Post-impressionist warm golden sunflowers in heavy oil paint" },
        { icon: "📐", label: "Bauhaus Shapes", prompt: "Mid-century Bauhaus geometric color-block canvas" },
        { icon: "🖤", label: "Monochrome Ink", prompt: "Minimalist Japanese sumi-e monochrome line drawing" },
      ];
    }
    // 3. Sofas & Seating
    if (name.includes("sofa") || name.includes("cushion") || name.includes("chair") || name.includes("couch") || name.includes("pillow")) {
      return [
        { icon: "🛋️", label: "Emerald Velvet", prompt: "Luxurious deep emerald green velvet with brass piping" },
        { icon: "🤎", label: "Cognac Leather", prompt: "Vintage hand-distressed Italian cognac leather" },
        { icon: "☁️", label: "Nordic Bouclé", prompt: "Minimalist Scandinavian cream bouclé fabric" },
        { icon: "🖤", label: "Matte Charcoal", prompt: "Modern architectural matte charcoal textured weave" },
      ];
    }
    // 4. Tables & Desks
    if (name.includes("table") || name.includes("desk") || name.includes("surface") || name.includes("stand")) {
      return [
        { icon: "🪵", label: "Live-Edge Walnut", prompt: "Solid organic live-edge American walnut wood" },
        { icon: "✨", label: "Calacatta Marble", prompt: "Polished Italian Calacatta gold marble with amber veining" },
        { icon: "🪨", label: "Brutalist Concrete", prompt: "Smooth hand-cast brutalist architectural concrete" },
        { icon: "🪞", label: "Smoked Glass", prompt: "Tempered smoked bronze glass with satin brass frame" },
      ];
    }
    // 5. Walls & Architectural Backdrops
    if (name.includes("wall") || name.includes("backdrop") || name.includes("partition")) {
      return [
        { icon: "🌿", label: "Sage Wood Slats", prompt: "Deep sage green accent wall with vertical oak acoustic slats" },
        { icon: "🏛️", label: "Venetian Plaster", prompt: "Textured warm beige Venetian limewash plaster" },
        { icon: "🧱", label: "Exposed Brick", prompt: "Rustic industrial exposed loft brick wall with warm mortar" },
        { icon: "🌌", label: "Midnight Blue", prompt: "Dramatic matte midnight blue accent wall with brass moldings" },
      ];
    }
    return [
      { icon: "✨", label: "Nordic Minimalist", prompt: "Light Scandinavian minimalist design with natural oak" },
      { icon: "🌿", label: "Japandi Organic", prompt: "Japandi aesthetic with wabi-sabi organic textures" },
      { icon: "🏛️", label: "Italian Luxury", prompt: "Contemporary Milanese luxury with brass accents and velvet" },
      { icon: "🖤", label: "Industrial Loft", prompt: "Urban industrial loft style with dark metals and leather" },
    ];
  };

  const testCategories = [
    { target: "Sofa", expectedFirst: "Emerald Velvet" },
    { target: "Armchair", expectedFirst: "Emerald Velvet" },
    { target: "Wall", expectedFirst: "Sage Wood Slats" },
    { target: "Backdrop Wall", expectedFirst: "Sage Wood Slats" },
    { target: "Coffee Table", expectedFirst: "Live-Edge Walnut" },
    { target: "Dining Table", expectedFirst: "Live-Edge Walnut" },
    { target: "Oil Canvas Art", expectedFirst: "Abstract Oil" },
    { target: "Wall Painting", expectedFirst: "Abstract Oil" },
    { target: "Coffee Mug", expectedFirst: "Terracotta Vase" },
    { target: "Ceramic Vase", expectedFirst: "Terracotta Vase" },
    { target: "Unknown Living Room", expectedFirst: "Nordic Minimalist" },
  ];

  testCategories.forEach(({ target, expectedFirst }) => {
    it(`generates 4 valid non-empty suggestions for "${target}"`, () => {
      const suggestions = getSuggestions(target);
      expect(suggestions).toHaveLength(4);
      expect(suggestions[0]?.label).toBe(expectedFirst);
      suggestions.forEach((s) => {
        expect(s.icon.length).toBeGreaterThan(0);
        expect(s.label.length).toBeGreaterThan(0);
        expect(s.prompt.length).toBeGreaterThan(10);
      });
    });
  });
});

describe("Responsive Viewport Dimensions & Layout Bounds", () => {
  const computeViewportLayout = (width: number, height: number) => {
    const isMobile = width < 768;
    const isTablet = width >= 768 && width < 1024;
    const isDesktop = width >= 1024;
    return {
      isMobile,
      isTablet,
      isDesktop,
      aspectRatio: width / height,
      popoverMode: isMobile ? "bottom-sheet" : "floating-glass",
      maxSheetHeightPx: isMobile ? Math.round(height * 0.48) : Math.round(height * 0.6),
    };
  };

  it("calculates correct bottom sheet layout for iPhone SE (320x568)", () => {
    const layout = computeViewportLayout(320, 568);
    expect(layout.isMobile).toBe(true);
    expect(layout.popoverMode).toBe("bottom-sheet");
    expect(layout.maxSheetHeightPx).toBe(273);
  });

  it("calculates correct bottom sheet layout for iPhone 14 Pro (393x852)", () => {
    const layout = computeViewportLayout(393, 852);
    expect(layout.isMobile).toBe(true);
    expect(layout.popoverMode).toBe("bottom-sheet");
    expect(layout.maxSheetHeightPx).toBe(409);
  });

  it("calculates correct bottom sheet layout for Samsung Galaxy S23 (412x915)", () => {
    const layout = computeViewportLayout(412, 915);
    expect(layout.isMobile).toBe(true);
    expect(layout.popoverMode).toBe("bottom-sheet");
    expect(layout.maxSheetHeightPx).toBe(439);
  });

  it("calculates correct floating glass layout for iPad Mini (768x1024)", () => {
    const layout = computeViewportLayout(768, 1024);
    expect(layout.isTablet).toBe(true);
    expect(layout.popoverMode).toBe("floating-glass");
  });

  it("calculates correct floating glass layout for MacBook Pro (1512x982)", () => {
    const layout = computeViewportLayout(1512, 982);
    expect(layout.isDesktop).toBe(true);
    expect(layout.popoverMode).toBe("floating-glass");
  });

  it("calculates correct floating glass layout for 4K Monitor (3840x2160)", () => {
    const layout = computeViewportLayout(3840, 2160);
    expect(layout.isDesktop).toBe(true);
    expect(layout.popoverMode).toBe("floating-glass");
  });
});

describe("3D Raycasting & Selection Hit Precision", () => {
  it("prioritizes tabletop items over large table surface", () => {
    const hits = [
      { name: "living_room_coffee_table", volume: 1.5, distance: 1.8 },
      { name: "ceramic_coffee_mug", volume: 0.002, distance: 1.82 },
    ];

    // Priority formula: smaller bounding volume gets prioritized when distance is almost identical
    const sorted = [...hits].sort((a, b) => {
      const distDiff = Math.abs(a.distance - b.distance);
      if (distDiff < 0.1) {
        return a.volume - b.volume;
      }
      return a.distance - b.distance;
    });

    expect(sorted[0]?.name).toBe("ceramic_coffee_mug");
  });

  it("prioritizes sofa cushions over floor", () => {
    const hits = [
      { name: "room_floor_parquet", volume: 12.0, distance: 2.1 },
      { name: "velvet_sofa_seat", volume: 0.8, distance: 2.14 },
    ];

    const sorted = [...hits].sort((a, b) => {
      const distDiff = Math.abs(a.distance - b.distance);
      if (distDiff < 0.1) return a.volume - b.volume;
      return a.distance - b.distance;
    });

    expect(sorted[0]?.name).toBe("velvet_sofa_seat");
  });
});
