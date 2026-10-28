/**
 * GharValue AI - Admin Console JavaScript
 * Handles modal inspection, mobile sidebar navigation, and history filtering.
 */

// Format numbers into Indian Rupee Currency representation
function formatRupeesINR(amount) {
  if (!amount && amount !== 0) return '₹0';
  const num = Math.round(Number(amount));
  const s = num.toString();
  if (s.length <= 3) return '₹' + s;
  const lastThree = s.substring(s.length - 3);
  let remaining = s.substring(0, s.length - 3);
  const parts = [];
  while (remaining.length > 2) {
    parts.unshift(remaining.substring(remaining.length - 2));
    remaining = remaining.substring(0, remaining.length - 2);
  }
  if (remaining.length > 0) parts.unshift(remaining);
  parts.push(lastThree);
  return '₹' + parts.join(',');
}

/**
 * Open Prediction Record Specification Modal
 * Called from Details / View button in dashboard and history tables.
 */
function openDetailModal(record) {
  const modal = document.getElementById('detailModal');
  const body = document.getElementById('modalDetailsBody');
  if (!modal || !body) return;

  const priceInr = Number(record.predicted_price_inr || 0);
  const priceLakhs = Number(record.predicted_price_lakhs || (priceInr / 100000));
  const superArea = Number(record.super_area_sqft || 1);
  const pricePerSqft = superArea > 0 ? Math.round(priceInr / superArea) : 0;

  body.innerHTML = `
    <!-- Valuation Highlight Banner -->
    <div style="background: linear-gradient(135deg, #1b4332 0%, #0d281e 100%); color: #ffffff; border-radius: var(--radius-lg); padding: 1.25rem 1.5rem; margin-bottom: 1.5rem; box-shadow: var(--shadow-sm);">
      <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 0.5rem;">
        <div>
          <span style="font-size: 0.72rem; color: #95d5b2; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Prediction Record ID</span>
          <div style="font-size: 1.15rem; font-weight: 800; font-family: monospace; color: #ffffff;">${record.prediction_id || 'N/A'}</div>
        </div>
        <div style="text-align: right;">
          <span class="badge badge-success" style="font-size: 0.72rem;"><i class="fa-solid fa-check"></i> ${record.status || 'Completed'}</span>
          <div style="font-size: 0.72rem; color: #95d5b2; margin-top: 0.25rem;">${record.timestamp || ''}</div>
        </div>
      </div>

      <div style="border-top: 1px solid rgba(255, 255, 255, 0.15); padding-top: 0.85rem; display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 1rem;">
        <div>
          <div style="font-size: 0.75rem; color: #d8f3dc; margin-bottom: 0.15rem;">Estimated Market Value</div>
          <div style="font-size: 1.75rem; font-weight: 800; color: #ffffff; line-height: 1.1;">${formatRupeesINR(priceInr)}</div>
        </div>
        <div style="display: flex; gap: 1rem;">
          <div style="background: rgba(255,255,255,0.1); padding: 0.4rem 0.75rem; border-radius: var(--radius-sm); text-align: center;">
            <div style="font-size: 0.65rem; color: #95d5b2;">IN LAKHS</div>
            <div style="font-size: 1.1rem; font-weight: 700;">₹${priceLakhs.toFixed(2)} L</div>
          </div>
          <div style="background: rgba(255,255,255,0.1); padding: 0.4rem 0.75rem; border-radius: var(--radius-sm); text-align: center;">
            <div style="font-size: 0.65rem; color: #95d5b2;">RATE / SQFT</div>
            <div style="font-size: 1.1rem; font-weight: 700;">₹${pricePerSqft.toLocaleString('en-IN')}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Property Layout & Location Grid -->
    <div style="margin-bottom: 1.25rem;">
      <h4 style="font-size: 0.82rem; font-weight: 700; color: var(--gray-600); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.65rem;">
        <i class="fa-solid fa-house" style="color: var(--primary-600); margin-right: 0.35rem;"></i> Layout & Physical Dimensions
      </h4>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 0.65rem;">
        <div style="background: #f8fafc; border: 1px solid var(--admin-border); border-radius: var(--radius-md); padding: 0.65rem 0.85rem;">
          <div style="font-size: 0.7rem; color: var(--gray-500);">City</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: var(--gray-900);">${record.city || 'N/A'}</div>
        </div>
        <div style="background: #f8fafc; border: 1px solid var(--admin-border); border-radius: var(--radius-md); padding: 0.65rem 0.85rem;">
          <div style="font-size: 0.7rem; color: var(--gray-500);">Locality Tier</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: var(--gray-900);">${record.locality_tier || 'N/A'}</div>
        </div>
        <div style="background: #f8fafc; border: 1px solid var(--admin-border); border-radius: var(--radius-md); padding: 0.65rem 0.85rem;">
          <div style="font-size: 0.7rem; color: var(--gray-500);">Configuration</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: var(--gray-900);">${record.bhk} BHK (${record.bathrooms} Baths)</div>
        </div>
        <div style="background: #f8fafc; border: 1px solid var(--admin-border); border-radius: var(--radius-md); padding: 0.65rem 0.85rem;">
          <div style="font-size: 0.7rem; color: var(--gray-500);">Super Built-Up Area</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: var(--gray-900);">${record.super_area_sqft} sqft</div>
        </div>
        <div style="background: #f8fafc; border: 1px solid var(--admin-border); border-radius: var(--radius-md); padding: 0.65rem 0.85rem;">
          <div style="font-size: 0.7rem; color: var(--gray-500);">Carpet Area</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: var(--gray-900);">${record.carpet_area_sqft} sqft</div>
        </div>
        <div style="background: #f8fafc; border: 1px solid var(--admin-border); border-radius: var(--radius-md); padding: 0.65rem 0.85rem;">
          <div style="font-size: 0.7rem; color: var(--gray-500);">Floor Level</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: var(--gray-900);">Floor ${record.floor_no} of ${record.total_floors}</div>
        </div>
        <div style="background: #f8fafc; border: 1px solid var(--admin-border); border-radius: var(--radius-md); padding: 0.65rem 0.85rem;">
          <div style="font-size: 0.7rem; color: var(--gray-500);">Property Age</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: var(--gray-900);">${record.property_age_years} Years</div>
        </div>
        <div style="background: #f8fafc; border: 1px solid var(--admin-border); border-radius: var(--radius-md); padding: 0.65rem 0.85rem;">
          <div style="font-size: 0.7rem; color: var(--gray-500);">Furnishing</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: var(--gray-900);">${record.furnishing || 'N/A'}</div>
        </div>
      </div>
    </div>

    <!-- Amenities & Building Features -->
    <div style="margin-bottom: 1.25rem;">
      <h4 style="font-size: 0.82rem; font-weight: 700; color: var(--gray-600); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.65rem;">
        <i class="fa-solid fa-gem" style="color: var(--primary-600); margin-right: 0.35rem;"></i> Building Amenities & Security
      </h4>
      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.65rem;">
        <div style="background: #f8fafc; border: 1px solid var(--admin-border); border-radius: var(--radius-md); padding: 0.65rem 0.85rem; display: flex; align-items: center; gap: 0.65rem;">
          <i class="fa-solid fa-square-parking" style="font-size: 1.2rem; color: ${Number(record.parking) > 0 ? '#10b981' : '#94a3b8'};"></i>
          <div>
            <div style="font-size: 0.7rem; color: var(--gray-500);">Parking</div>
            <div style="font-size: 0.85rem; font-weight: 700;">${Number(record.parking) > 0 ? 'Available' : 'None'}</div>
          </div>
        </div>

        <div style="background: #f8fafc; border: 1px solid var(--admin-border); border-radius: var(--radius-md); padding: 0.65rem 0.85rem; display: flex; align-items: center; gap: 0.65rem;">
          <i class="fa-solid fa-elevator" style="font-size: 1.2rem; color: ${Number(record.lift) > 0 ? '#10b981' : '#94a3b8'};"></i>
          <div>
            <div style="font-size: 0.7rem; color: var(--gray-500);">Passenger Lift</div>
            <div style="font-size: 0.85rem; font-weight: 700;">${Number(record.lift) > 0 ? 'Equipped' : 'No Lift'}</div>
          </div>
        </div>

        <div style="background: #f8fafc; border: 1px solid var(--admin-border); border-radius: var(--radius-md); padding: 0.65rem 0.85rem; display: flex; align-items: center; gap: 0.65rem;">
          <i class="fa-solid fa-shield-halved" style="font-size: 1.2rem; color: ${Number(record.gated_society) > 0 ? '#10b981' : '#94a3b8'};"></i>
          <div>
            <div style="font-size: 0.7rem; color: var(--gray-500);">Gated Security</div>
            <div style="font-size: 0.85rem; font-weight: 700;">${Number(record.gated_society) > 0 ? 'Gated Society' : 'Independent'}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Geospatial Proximity & Crime Index -->
    <div>
      <h4 style="font-size: 0.82rem; font-weight: 700; color: var(--gray-600); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.65rem;">
        <i class="fa-solid fa-location-dot" style="color: var(--primary-600); margin-right: 0.35rem;"></i> Geospatial Proximity & Neighborhood Safety
      </h4>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 0.65rem;">
        <div style="background: #f8fafc; border: 1px solid var(--admin-border); border-radius: var(--radius-md); padding: 0.65rem 0.85rem;">
          <div style="font-size: 0.7rem; color: var(--gray-500);">Distance to Metro</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: var(--gray-900);">${record.distance_to_metro_km} km</div>
        </div>
        <div style="background: #f8fafc; border: 1px solid var(--admin-border); border-radius: var(--radius-md); padding: 0.65rem 0.85rem;">
          <div style="font-size: 0.7rem; color: var(--gray-500);">City Center</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: var(--gray-900);">${record.distance_to_citycenter_km} km</div>
        </div>
        <div style="background: #f8fafc; border: 1px solid var(--admin-border); border-radius: var(--radius-md); padding: 0.65rem 0.85rem;">
          <div style="font-size: 0.7rem; color: var(--gray-500);">Nearby School</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: var(--gray-900);">${record.nearby_school_km} km</div>
        </div>
        <div style="background: #f8fafc; border: 1px solid var(--admin-border); border-radius: var(--radius-md); padding: 0.65rem 0.85rem;">
          <div style="font-size: 0.7rem; color: var(--gray-500);">Nearby Hospital</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: var(--gray-900);">${record.nearby_hospital_km} km</div>
        </div>
        <div style="background: #f8fafc; border: 1px solid var(--admin-border); border-radius: var(--radius-md); padding: 0.65rem 0.85rem;">
          <div style="font-size: 0.7rem; color: var(--gray-500);">Crime Rate Index</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: ${Number(record.crime_rate_index) < 30 ? '#15803d' : '#b45309'};">
            ${record.crime_rate_index} / 100
          </div>
        </div>
      </div>
    </div>
  `;

  modal.classList.add('open');
}

// Close Modal
function closeDetailModal() {
  const modal = document.getElementById('detailModal');
  if (modal) {
    modal.classList.remove('open');
  }
}

// Bind functions to window so inline onclick handlers always work
window.openDetailModal = openDetailModal;
window.closeDetailModal = closeDetailModal;

document.addEventListener('DOMContentLoaded', function () {
  // 1. Mobile Sidebar Toggle
  const toggleBtn = document.getElementById('sidebarToggle');
  const sidebar = document.getElementById('adminSidebar');
  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener('click', function () {
      sidebar.classList.toggle('open');
    });
  }

  // 2. Modal Close Event Handlers
  document.querySelectorAll('.modal-close-trigger').forEach(btn => {
    btn.addEventListener('click', closeDetailModal);
  });

  const modal = document.getElementById('detailModal');
  if (modal) {
    modal.addEventListener('click', function (e) {
      if (e.target === modal) {
        closeDetailModal();
      }
    });
  }

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') {
      closeDetailModal();
    }
  });

  // 3. Prediction History Search & City Filter
  const searchInput = document.getElementById('historySearch');
  const cityFilter = document.getElementById('historyCityFilter');
  const table = document.getElementById('historyTable');

  if (table && (searchInput || cityFilter)) {
    function filterHistoryTable() {
      const query = (searchInput ? searchInput.value : '').toLowerCase().trim();
      const selectedCity = cityFilter ? cityFilter.value.toLowerCase().trim() : '';

      const rows = table.querySelectorAll('tbody tr');
      rows.forEach(row => {
        // Skip empty row if present
        if (row.children.length === 1) return;

        const rowText = row.innerText.toLowerCase();
        const rowCity = (row.getAttribute('data-city') || '').toLowerCase();

        const matchesQuery = !query || rowText.includes(query);
        const matchesCity = !selectedCity || rowCity === selectedCity;

        row.style.display = (matchesQuery && matchesCity) ? '' : 'none';
      });
    }

    if (searchInput) searchInput.addEventListener('input', filterHistoryTable);
    if (cityFilter) cityFilter.addEventListener('change', filterHistoryTable);
  }
});
