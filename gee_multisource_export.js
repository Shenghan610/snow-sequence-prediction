// Google Earth Engine Code Editor script.
// Official data sources used here match the project document:
// - MODIS/061/MOD10A1 snow cover
// - ECMWF/ERA5_LAND/DAILY_AGGR weather
// - USGS/SRTMGL1_003 terrain
//
// Paste into https://code.earthengine.google.com and run after signing in.

var studyRegion = ee.Geometry.Rectangle([78.3944, 29.6778, 86.1975, 35.7153]);
var startDate = '2020-01-01';
var endDate = '2024-12-31';

var snowCollection = ee.ImageCollection('MODIS/061/MOD10A1')
  .filter(ee.Filter.date(startDate, ee.Date(endDate).advance(1, 'day')))
  .select('NDSI_Snow_Cover')
  .map(function(img) {
    var snowCover = img.select('NDSI_Snow_Cover');
    var mask = snowCover.gte(0).and(snowCover.lte(100));
    return img.updateMask(mask)
      .clip(studyRegion)
      .set('system:time_start', img.get('system:time_start'));
  });

var dem = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(studyRegion);
var slope = ee.Terrain.slope(dem);
var aspect = ee.Terrain.aspect(dem);
var aspectRad = aspect.multiply(Math.PI).divide(180);
var slopeRad = slope.multiply(Math.PI).divide(180);
var radiationIndex = aspectRad.cos().multiply(slopeRad.sin());
var terrainFeatures = dem.rename('elevation')
  .addBands(slope.rename('slope'))
  .addBands(aspect.rename('aspect'))
  .addBands(radiationIndex.rename('terrain_radiation_index'));

var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
  .filter(ee.Filter.date(startDate, ee.Date(endDate).advance(1, 'day')))
  .select([
    'temperature_2m',
    'dewpoint_temperature_2m',
    'total_precipitation_sum',
    'snowfall_sum',
    'u_component_of_wind_10m',
    'v_component_of_wind_10m',
    'surface_net_solar_radiation_sum',
    'surface_net_thermal_radiation_sum'
  ])
  .map(function(img) {
    var u = img.select('u_component_of_wind_10m');
    var v = img.select('v_component_of_wind_10m');
    var windSpeed = u.pow(2).add(v.pow(2)).sqrt().rename('wind_speed');
    var tempC = img.select('temperature_2m').subtract(273.15).rename('temp_celsius');
    var dewC = img.select('dewpoint_temperature_2m').subtract(273.15).rename('dew_celsius');
    return img.addBands(windSpeed)
      .addBands(tempC)
      .addBands(dewC)
      .clip(studyRegion);
  });

function createFusedImage(snowImage) {
  var date = ee.Date(snowImage.get('system:time_start'));
  var snow = snowImage.select('NDSI_Snow_Cover').toFloat();
  var meteo = era5.filterDate(date, date.advance(1, 'day')).first()
    .resample('bilinear')
    .reproject({crs: snow.projection(), scale: 500})
    .toFloat();
  var terrain = terrainFeatures
    .resample('bilinear')
    .reproject({crs: snow.projection(), scale: 500})
    .toFloat();
  return snow.addBands(meteo)
    .addBands(terrain)
    .set('system:time_start', date.millis())
    .set('date', date.format('YYYY-MM-dd'));
}

var fusedCollection = snowCollection.map(createFusedImage);
print('Fused bands', fusedCollection.first().bandNames());
print('Fused collection size', fusedCollection.size());

fusedCollection.aggregate_array('system:time_start').evaluate(function(timeStarts) {
  timeStarts.forEach(function(timeStart) {
    var img = fusedCollection.filter(ee.Filter.eq('system:time_start', timeStart)).first();
    var dateStr = ee.Date(timeStart).format('YYYY-MM-dd').getInfo();
    Export.image.toDrive({
      image: img,
      description: 'MultimodalSnow_' + dateStr,
      folder: 'GEE_SnowPrediction_Multimodal',
      region: studyRegion,
      scale: 500,
      crs: 'EPSG:4326',
      maxPixels: 1e13,
      fileFormat: 'GeoTIFF'
    });
  });
});

Map.setCenter(82.2959, 32.6965, 7);
Map.addLayer(snowCollection.first(), {min: 0, max: 100, palette: ['black', '0dffff', '0524ff', 'ffffff']}, 'MODIS Snow');
Map.addLayer(dem, {min: 3000, max: 6500, palette: ['green', 'yellow', 'brown', 'white']}, 'SRTM Elevation');
