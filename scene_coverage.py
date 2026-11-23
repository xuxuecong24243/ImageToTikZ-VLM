"""Read-only evidence checks. Expected content belongs to evaluation data, never prompts.

Literal matches and declared graph endpoints are evidence, not OCR/semantic proof.
Without reference expectations, specific omissions or geometric fidelity are unknown.
"""
from collections import defaultdict
from structured_tikz.schema import NODE_TYPES
from scene_structure import inspect_structure


def evaluate_coverage(scene, expectations=None, geometry=None):
    report={'format':inspect_structure(scene),'recognised':[],'missing':[],
            'unsupported':[],'unknown':[],'quality_warning':False,
            'geometry':{'fidelity':'unknown','note':'Candidate geometry is not ground truth or OCR.'},
            'topology':{'correctness':'unknown'},'content':{'complete':None},
            'limitations':['Literal text presence does not prove correct placement or technical meaning.',
                          'Declared node references are proposals, not visually verified connections.',
                          'Schema validity and coverage do not establish visual similarity.']}
    data=scene if isinstance(scene,dict) else {}
    def records(key):
        value=data.get(key,[])
        return [x for x in value[:500] if isinstance(x,dict)] if isinstance(value,list) else []
    nodes,labels,edges=records('nodes'),records('labels'),records('edges')
    node_ids={n.get('id') for n in nodes if isinstance(n.get('id'),str)}
    def text(item):
        if item.get('mode')=='rich' and isinstance(item.get('segments'),list):
            return ''.join(s.get('content','') for s in item['segments'] if isinstance(s,dict) and isinstance(s.get('content'),str))
        return item.get('text','') if isinstance(item.get('text',''),str) else ''
    rendered_text={item['id']:text(item) for item in nodes+labels if isinstance(item.get('id'),str)}
    report['counts']={'nodes':len(nodes),'labels':len(labels),'edges':len(edges),
                      'nonempty_text_objects':sum(bool(t.strip()) for t in rendered_text.values())}
    if isinstance(geometry,dict):
        report['geometry']['candidate_counts']={k:len(geometry[k]) for k in ('rectangles','lines','regions','text_regions') if isinstance(geometry.get(k),list)}
        report['geometry']['text_candidate_count_is_not_required_text_nodes']=True
    for node in nodes:
        if not isinstance(node.get('type'),str) or node['type'] not in NODE_TYPES:
            report['unsupported'].append({'id':node.get('id'),'type':node.get('type'),'reason':'unsupported_native_primitive'})
    metadata=data.get('metadata',{})
    if isinstance(metadata,dict):
        for key in ('unsupported_elements','unreadable_regions','omitted_elements','pending_nodes','pending_labels','pending_parameters','pending_edges'):
            if metadata.get(key):report['unknown'].append({'kind':key,'evidence':metadata[key],'source':'model_self_report_not_verified'})
    if expectations is None:
        report['unknown'].append({'kind':'reference_unavailable','reason':'No expected content; cannot identify specific omissions from counts.'})
        report['quality_warning']=bool(report['unsupported'] or (isinstance(metadata,dict) and any(metadata.get(k) for k in ('unsupported_elements','omitted_elements','pending_nodes','pending_labels','pending_parameters','pending_edges'))))
        return report
    if not isinstance(expectations,dict) or not isinstance(expectations.get('content',[]),list):
        raise ValueError('Coverage expectations must be an object with content/relations arrays.')
    found={}; owners={}
    by_id={x.get('id'):x for x in nodes+labels if isinstance(x.get('id'),str)}
    for expected in expectations.get('content',[]):
        if not isinstance(expected,dict) or not isinstance(expected.get('id'),str):
            raise ValueError('Each content expectation needs an id.')
        terms=expected.get('all_of',[]);exact=expected.get('exact',[])
        if not isinstance(terms,list) or not isinstance(exact,list):
            raise ValueError('Literal expectations must be arrays.')
        if (not terms and not exact) or any(not isinstance(t,str) or not t for t in terms+exact):
            raise ValueError('Content expectations require nonempty literal all_of/exact terms.')
        evidence={term:[identifier for identifier,value in rendered_text.items()
                        if (value.strip()==term if term in exact else term in value)] for term in terms+exact}
        matched=all(evidence.values());found[expected['id']]=matched
        item={'id':expected['id'],'evidence':evidence,'basis':'literal_presence_only'}
        report['recognised' if matched else 'missing'].append(item)
        mapped=set()
        for identifiers in evidence.values():
            for identifier in identifiers:
                if identifier in node_ids:mapped.add(identifier)
                else:
                    meta=by_id[identifier].get('metadata',{})
                    owner=meta.get('node_id') if isinstance(meta,dict) else None
                    if isinstance(owner,str) and owner in node_ids:mapped.add(owner)
        owners[expected['id']]=mapped
    graph=defaultdict(set);ambiguous_paths=False
    for edge in edges:
        meta=edge.get('metadata',{}) if isinstance(edge.get('metadata',{}),dict) else {}
        source=edge.get('from',meta.get('source_id'));target=edge.get('to',meta.get('target_id'))
        if not isinstance(source,str) or not isinstance(target,str) or source not in node_ids or target not in node_ids:
            ambiguous_paths=True;continue
        arrow=edge.get('arrow','none')
        if arrow in ('end','both'):graph[source].add(target)
        if arrow in ('start','both'):graph[target].add(source)
    def reachable(start,ends):
        # Require at least one directed edge; shared containers do not establish a flow.
        todo=list(graph[start]);seen=set()
        while todo:
            item=todo.pop()
            if item in ends:return True
            if item not in seen:seen.add(item);todo.extend(graph[item]-seen)
        return False
    relations=expectations.get('relations',[])
    if not isinstance(relations,list):raise ValueError('relations must be an array.')
    relation_results=[]
    for relation in relations:
        if not isinstance(relation,dict) or not all(isinstance(relation.get(k),str) for k in ('id','source','target')):
            raise ValueError('Relations require id/source/target expectation IDs.')
        source,target=relation['source'],relation['target']
        if source not in found or target not in found:raise ValueError('Relation references an unknown content expectation.')
        if not found[source] or not found[target] or not edges:
            state='missing';reason='expected_endpoints_or_edges_absent'
        elif len(owners[source])!=1 or len(owners[target])!=1 or ambiguous_paths or owners[source]==owners[target]:
            state='unknown';reason='edge_endpoints_or_label_ownership_not_explicit'
        elif any((bool(graph[start] & owners[target]) if relation.get('direct') else reachable(start,owners[target])) for start in owners[source]):
            state='recognised';reason='directed_path_in_declared_graph_only'
        else:
            state='missing';reason='no_directed_path_in_declared_graph'
        if state=='recognised' and relation.get('branch_label'):
            state='unknown';reason='branch_label_to_edge_association_requires_independent_review'
        result=dict(relation,status=state,reason=reason)
        relation_results.append(result);report[state].append(result)
    for kind in expectations.get('native_primitives',[]):
        if kind not in NODE_TYPES:
            report['unsupported'].append({'type':kind,'reason':'expected_native_shape_not_supported_by_current_schema'})
    report['topology']={'correctness':'not_visually_verified','relations':relation_results,
                        'unresolved_explicit_paths':ambiguous_paths}
    report['content']={'expected_items':len(found),'literal_items_found':sum(found.values()),
                        'complete':not report['missing'] and not report['unknown'] and not report['unsupported'],
                        'scope':'Only the supplied offline expectations; not proof of complete image reconstruction.'}
    report['quality_warning']=bool(report['missing'] or report['unsupported'] or report['unknown'])
    return report
