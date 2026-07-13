// The DatacenterProfile POSTed to /run: five closed dropdown selections that
// the backend resolves into a full TargetSpec + CriticWeights server-side
// (schemas/datacenter_profile.py). No free-text spec input exists — the
// option values below must match the backend's Literal types exactly.

export interface DatacenterProfile {
  cooling_method: 'single_phase_immersion' | 'direct_to_chip' | 'rear_door_heat_exchanger'
  rack_density: 'standard' | 'high_density' | 'ultra_high_density'
  climate_zone: 'temperate' | 'hot_humid' | 'cold'
  regulatory_region: 'us' | 'eu' | 'apac' | 'global'
  optimization_priority: 'performance' | 'cost' | 'compliance' | 'lifespan'
}

export interface ProfileField {
  key: keyof DatacenterProfile
  label: string
  options: { value: string; label: string }[]
}

export const PROFILE_FIELDS: ProfileField[] = [
  {
    key: 'cooling_method',
    label: 'Cooling method',
    options: [
      { value: 'single_phase_immersion', label: 'Single-phase immersion' },
      { value: 'direct_to_chip', label: 'Direct-to-chip' },
      { value: 'rear_door_heat_exchanger', label: 'Rear-door heat exchanger' },
    ],
  },
  {
    key: 'rack_density',
    label: 'Rack density',
    options: [
      { value: 'standard', label: 'Standard' },
      { value: 'high_density', label: 'High density' },
      { value: 'ultra_high_density', label: 'Ultra-high density' },
    ],
  },
  {
    key: 'climate_zone',
    label: 'Climate zone',
    options: [
      { value: 'temperate', label: 'Temperate' },
      { value: 'hot_humid', label: 'Hot & humid' },
      { value: 'cold', label: 'Cold' },
    ],
  },
  {
    key: 'regulatory_region',
    label: 'Regulatory region',
    options: [
      { value: 'us', label: 'United States (TSCA)' },
      { value: 'eu', label: 'European Union (REACH)' },
      { value: 'apac', label: 'APAC (regional baseline)' },
      { value: 'global', label: 'Global (strictest combined)' },
    ],
  },
  {
    key: 'optimization_priority',
    label: 'Optimization priority',
    options: [
      { value: 'performance', label: 'Performance' },
      { value: 'cost', label: 'Cost' },
      { value: 'compliance', label: 'Compliance' },
      { value: 'lifespan', label: 'Lifespan' },
    ],
  },
]

export const DEFAULT_PROFILE: DatacenterProfile = {
  cooling_method: 'single_phase_immersion',
  rack_density: 'high_density',
  climate_zone: 'temperate',
  regulatory_region: 'global',
  optimization_priority: 'performance',
}
