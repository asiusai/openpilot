"""Generic text proposals; the existing target verifier decides every token.

These hand-written phrases use no recorded predictions or future inputs.
They are a fallback after observed text history. They never directly become
model output. Different verifier shapes can still change numerical results.
"""
from types import MethodType

PHRASES = (
  '<NOMINAL>',
  'Follow the road.',
  'Continue straight to follow our route.',
  'Continue driving straight along the road.',
  'Maintain the current lane and speed.',
  'Slow down for the vehicle ahead.',
  'Decelerate and stop at the red traffic light.',
  'Stop and yield to crossing traffic.',
  'Yield to pedestrians crossing the road.',
  'Proceed through the intersection as the traffic light is green.',
  'Turn left at the intersection to follow our route.',
  'Turn right at the intersection to follow our route.',
  'Decelerate and turn left at the intersection to follow our route.',
  'Decelerate and turn right at the intersection to follow our route.',
  'Follow the curve of the road to the left.',
  'Follow the curve of the road to the right.',
  'Enter the roundabout and follow our route.',
  'Take the exit from the roundabout to follow our route.',
)

def install(decoder, tokenizer, anchor):
  bank = tuple(tuple(tokenizer.encode(text+'\n<|traj_future_start|>', add_special_tokens=False)) for text in PHRASES)
  if any(not tokens or tokens[-1] != anchor or anchor in tokens[:-1] or len(tokens)>64 for tokens in bank):
    raise ValueError('Each proposal must have one terminal trajectory marker and fit the verifier budget')
  original = decoder.continuation
  statistics = {'fallback_proposals':0, 'fallback_proposed_tokens':0}
  def continuation(self, context, limit):
    known = original(context, limit)
    if known or limit <= 0:
      return known
    for width in range(min(8,len(context)),0,-1):
      suffix = tuple(context[-width:])
      for previous in bank:
        for end in range(len(previous)-1,width-1,-1):
          if previous[end-width:end] == suffix:
            proposal = list(previous[end:end+limit])
            statistics['fallback_proposals'] += 1
            statistics['fallback_proposed_tokens'] += len(proposal)
            return proposal
    return []
  decoder.continuation = MethodType(continuation,decoder)
  decoder.phrase_draft_statistics = statistics
  return {'phrases':len(bank), 'tokens':sum(map(len,bank)),
          'source':'Hand-written generic driving language; no recorded model outputs',
          'acceptance':'Unchanged current-target verification; history continuations take priority'}
