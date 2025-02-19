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

            // If this node is already active, clear all highlighting and remove doc links.
            if (d3.select(this).classed("active")) {
              d3.selectAll("g.node").classed("active", false).classed("dim", false);
              d3.selectAll("g.edge").classed("highlight", false);
              d3.selectAll("svg a.doc-link").remove();
              return;
            }

            // Clear previous state.
            d3.selectAll("g.node").classed("active", false).classed("dim", false);
            d3.selectAll("g.edge").classed("highlight", false);
            d3.selectAll("svg a.doc-link").remove();

            // Mark the clicked node as active.
            d3.select(this).classed("active", true);

            // Recursive function to highlight all ancestor nodes.
            function highlightParents(label, visited) {
              if (visited.indexOf(label) >= 0) return;
              visited.push(label);
              d3.selectAll("g.edge").each(function() {
                var edgeTitle = d3.select(this).select("title").text().trim();
                var parts = edgeTitle.split("->");
                if (parts.length === 2) {
                  var source = parts[0].trim();
                  var target = parts[1].trim();
                  if (target === label) {
                    // Highlight this edge.
                    d3.select(this).classed("highlight", true);
                    // Mark the source node as active.
                    d3.selectAll("g.node").filter(function() {
                      var nodeLabel = d3.select(this).select("title").text().trim();
                      return nodeLabel === source;
                    }).classed("active", true);
                    // Recurse to find parent's parent.
                    highlightParents(source, visited);
                  }
                }
              });
            }
            highlightParents(clickedLabel, []);

            // Dim nodes that are not active.
            d3.selectAll("g.node").filter(function() {
              return !d3.select(this).classed("active");
            }).classed("dim", true);

            // (Optional) Dim edges that are not highlighted.
            d3.selectAll("g.edge").filter(function() {
              return !d3.select(this).classed("highlight");
            }).classed("dim", true);

            // --- Append the "doc" link next to the clicked node ---
            // Use a lookup table or default URL based on the node's label.
            // For simplicity, we use a default URL:
            var docLinkURL = "http://localhost:8000/index.html#" + encodeURIComponent(clickedLabel);

            // Compute the absolute position of the node's right edge with a constant offset.
            var bbox = this.getBBox();
            var ctm = this.getCTM();
            var scale = Math.sqrt(ctm.a * ctm.a + ctm.b * ctm.b);
            var desiredOffset = 5; // constant 5-pixel gap in screen space
            var offsetInLocal = desiredOffset / scale;
            var pt = this.ownerSVGElement.createSVGPoint();
            pt.x = bbox.x + bbox.width + offsetInLocal;
            pt.y = bbox.y + bbox.height / 2;
            var globalPt = pt.matrixTransform(ctm);

            // Append the doc link to the top-level SVG container.
            d3.select(this.ownerSVGElement)
              .append("svg:a")
              .attr("class", "doc-link")
              .attr("xlink:href", docLinkURL)
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
