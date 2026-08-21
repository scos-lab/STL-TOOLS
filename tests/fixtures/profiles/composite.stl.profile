# Test fixture: profile manifest bundling the three namespaced fixture schemas.
# Included names resolve to <name>.stl.schema next to this manifest.
profile CompositeFixture v1.0 {
    include: [core, delivery, operations]
}
