document.addEventListener("DOMContentLoaded", function() {

  fetch("/_static/graph.dot?" + new Date().getTime())
    .then(function(response) {
      if (!response.ok) {
        throw new Error("Network response was not ok: " + response.statusText);
      }
      return response.text();
    })
    .then(function(dot) {
      d3.select("#graph").graphviz().renderDot(dot)
        .on("end", function() {
          // Attach click event to every node.
          d3.selectAll("g.node").on("click", function(event) {
            // Retrieve the node's label from its <title>.
            var clickedLabel = d3.select(this).select("title").text().trim();

            // Toggle: if this node is already active, clear all styling and remove doc links.
            if (d3.select(this).classed("active")) {
              d3.selectAll("g.node").classed("active", false).classed("dim", false);
              d3.selectAll("g.edge").classed("active", false).classed("dim", false);
              d3.selectAll("svg a.doc-link").remove();
              return;
            }

            // Clear previous state.
            d3.selectAll("g.node").classed("active", false).classed("dim", false);
            d3.selectAll("g.edge").classed("active", false).classed("dim", false);
            d3.selectAll("svg a.doc-link").remove();

            // Mark the clicked node as active.
            d3.select(this).classed("active", true);

            // Dim every other node.
            d3.selectAll("g.node").filter(function() {
              var nodeLabel = d3.select(this).select("title").text().trim();
              return nodeLabel !== clickedLabel;
            }).classed("dim", true);

            // Dim edges as needed.
            d3.selectAll("g.edge").classed("dim", true);

            // Define a lookup table: keys are node labels, values are documentation URLs.
            var docLookup = {
              "v_local": "http://localhost:8000/index.html#module-jrystal.pseudopotential.local",
              "coeff": "http://localhost:8000/index.html#jrystal._src.pw.density_grid_reciprocal",
            };

            // Look up the documentation URL using the lookup table.
            var nodeDoc = docLookup[clickedLabel] ||
                          "http://localhost:8000/index.html#" + encodeURIComponent(clickedLabel);

            // Compute the absolute position of the node's right edge with a constant pixel offset.
            var bbox = this.getBBox();
            var ctm = this.getCTM();
            var scale = Math.sqrt(ctm.a * ctm.a + ctm.b * ctm.b);
            var desiredOffset = 5; // constant 5-pixel gap
            var offsetInLocal = desiredOffset / scale;
            var pt = this.ownerSVGElement.createSVGPoint();
            pt.x = bbox.x + bbox.width + offsetInLocal;
            pt.y = bbox.y + bbox.height / 2;
            var globalPt = pt.matrixTransform(ctm);

            // Append the doc link to the top-level SVG container at the computed position.
            d3.select(this.ownerSVGElement)
              .append("svg:a")
              .attr("class", "doc-link")
              .attr("xlink:href", nodeDoc)
              .attr("target", "_blank")
              .append("svg:text")
              .text("doc")
              .attr("x", globalPt.x)
              .attr("y", globalPt.y);
          });
        });
    })
    .catch(function(error) {
      console.error("Error fetching DOT file:", error);
    });
});
