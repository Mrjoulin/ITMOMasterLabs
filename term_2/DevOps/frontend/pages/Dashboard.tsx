import React, { useState, useRef, useEffect } from 'react';
import { AppData, Area, SearchResult, Task, Note } from '../types';
import { api } from '../services/api';

interface DashboardProps {
  data: AppData | null;
  onNavigate: (page: string) => void;
  onNewTask: (areaId?: number | null) => void;
  onNewNote: (areaId?: number | null) => void;
  onOpenTask: (task: Task) => void;
  onOpenNote: (note: Note) => void;
}

const getTimeOfDayGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 4) return 'Good evening';
    if (hour < 12) return 'Good morning';
    if (hour < 18) return 'Good afternoon';
    return 'Good evening';
};

const Dashboard: React.FC<DashboardProps> = ({ data, onNavigate, onNewTask, onNewNote, onOpenTask, onOpenNote }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearchLoading, setIsSearchLoading] = useState(false);
  const [noSearchResults, setNoSearchResults] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = setTimeout(async () => {
      if (searchQuery.length > 1) {
        setIsSearchLoading(true);
        try {
          const results = await api.search(searchQuery, 'all');
          setSearchResults(results);
          setNoSearchResults(results.length === 0);
        } catch (error) {
          console.error("Search failed:", error);
          setSearchResults([]);
          setNoSearchResults(true);
        } finally {
          setIsSearchLoading(false);
        }
      } else {
        setSearchResults([]);
        setNoSearchResults(false);
      }
    }, 300); // 300ms debounce

    return () => {
      clearTimeout(handler);
    };
  }, [searchQuery]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
        if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
            setSearchResults([]);
            setNoSearchResults(false);
            setSearchQuery('');
        }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleOpenResult = (result: SearchResult) => {
    if (!data) return;
    if (result.type === 'task') {
      const task = data.tasks.find(t => t.id === result.id);
      if (task) {
        onOpenTask(task);
      }
    } else if (result.type === 'note') {
      const note = data.notes.find(n => n.id === result.id);
      if (note) {
        onOpenNote(note);
      }
    }
    setSearchResults([]);
    setNoSearchResults(false);
    setSearchQuery('');
  };

  if (!data) return <div className="p-8 text-center">Loading dashboard...</div>;

  const activeTasks = data.tasks.filter(t => !t.completed).length;
  const totalNotes = data.notes.length;

  // Calculate most active area
  let mostActiveArea: Area | null = null;
  let maxActivity = -1;

  data.areas.forEach(area => {
    const tasksInArea = data.tasks.filter(t => t.area_id === area.id && !t.completed).length;
    const notesInArea = data.notes.filter(n => n.area_id === area.id).length;
    const activity = tasksInArea + notesInArea;

    if (activity > maxActivity) {
      maxActivity = activity;
      mostActiveArea = area;
    }
  });

  return (
    <div className="px-4 sm:px-6 lg:px-8 py-8 w-full h-full overflow-y-auto">
      {/* Welcome */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white">{getTimeOfDayGreeting()}, {data.user.full_name.split(' ')[0]}</h1>
          <p className="mt-1 text-slate-500 dark:text-slate-400">Here is your daily brief for <span className="font-medium text-primary">{new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}</span>.</p>
        </div>
        <div className="flex items-center gap-3 relative" ref={searchRef}>
             <div className="relative">
                <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400">
                    <span className="material-icons text-lg">search</span>
                </span>
                <input
                    type="text"
                    placeholder="Search all..."
                    className="w-48 pl-10 pr-3 py-2.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-neutral-surface-dark text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-shadow shadow-sm outline-none"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                />
                {isSearchLoading && (
                    <span className="absolute inset-y-0 right-0 flex items-center pr-3">
                    <span className="animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-primary"></span>
                    </span>
                )}
            </div>
            {searchResults.length > 0 && searchQuery.length > 1 && (
                <div className="absolute left-0 right-0 top-full mt-2 bg-white dark:bg-slate-800 rounded-lg shadow-xl border border-slate-200 dark:border-slate-700 z-30 max-h-80 overflow-y-auto animate-fade-in-down">
                    {searchResults.map((result) => (
                    <button
                        key={result.type + result.id}
                        onClick={() => handleOpenResult(result)}
                        className="w-full text-left p-3 hover:bg-slate-50 dark:hover:bg-slate-700 flex flex-col transition-colors border-b border-slate-100 dark:border-slate-700 last:border-b-0"
                    >
                        <span className="text-xs text-slate-500 dark:text-slate-400 uppercase font-semibold">{result.type}</span>
                        <h4 className="font-semibold text-slate-900 dark:text-white truncate">{result.title}</h4>
                        {result.type === 'task' && result.description && (
                        <p className="text-sm text-slate-500 dark:text-slate-400 line-clamp-1">{result.description}</p>
                        )}
                        {result.type === 'note' && result.content && (
                        <p className="text-sm text-slate-500 dark:text-slate-400 line-clamp-1">{result.content}</p>
                        )}
                    </button>
                    ))}
                </div>
            )}
            {searchQuery.length > 1 && !isSearchLoading && noSearchResults && (
                <div className="absolute left-0 right-0 top-full mt-2 bg-white dark:bg-slate-800 rounded-lg shadow-xl border border-slate-200 dark:border-slate-700 z-30 p-3 text-center text-sm text-slate-500 animate-fade-in-down">
                    No results found.
                </div>
            )}
             <button onClick={() => onNewTask(null)} className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium shadow-sm transition-all bg-primary text-white hover:bg-primary-hover shadow-primary/30">
                <span className="material-icons text-lg text-white">add_task</span>
                New Task
             </button>
             <button onClick={() => onNewNote(null)} className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium shadow-sm transition-all bg-white dark:bg-neutral-surface-dark border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:border-primary text-primary">
                <span className="material-icons text-lg text-primary">note_add</span>
                New Note
             </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
        <div onClick={() => onNavigate('tasks')} className="bg-white dark:bg-neutral-surface-dark rounded-xl p-6 shadow-sm border border-slate-200 dark:border-slate-700 relative overflow-hidden group hover:shadow-md transition-all cursor-pointer">
            <div className="absolute right-0 top-0 h-full w-1 bg-primary"></div>
            <div className="flex justify-between items-start mb-4">
                <div className="p-2 bg-primary/10 rounded-lg text-primary"><span className="material-icons">check_circle</span></div>
            </div>
            <div className="flex flex-col">
                <span className="text-4xl font-bold text-slate-900 dark:text-white mb-1">{activeTasks}</span>
                <span className="text-sm font-medium text-slate-500 dark:text-slate-400">Active Tasks</span>
            </div>
        </div>

         <div onClick={() => onNavigate('notes')} className="bg-white dark:bg-neutral-surface-dark rounded-xl p-6 shadow-sm border border-slate-200 dark:border-slate-700 relative overflow-hidden group hover:shadow-md transition-all cursor-pointer">
            <div className="absolute right-0 top-0 h-full w-1 bg-slate-300 dark:bg-slate-600"></div>
            <div className="flex justify-between items-start mb-4">
                <div className="p-2 bg-slate-100 dark:bg-slate-700 rounded-lg text-slate-600 dark:text-slate-300"><span className="material-icons">description</span></div>
            </div>
            <div className="flex flex-col">
                <span className="text-4xl font-bold text-slate-900 dark:text-white mb-1">{totalNotes}</span>
                <span className="text-sm font-medium text-slate-500 dark:text-slate-400">Total Notes</span>
            </div>
        </div>

         <div onClick={() => mostActiveArea && onNavigate(`area:${mostActiveArea.id}`)} className="bg-white dark:bg-neutral-surface-dark rounded-xl p-6 shadow-sm border border-slate-200 dark:border-slate-700 relative overflow-hidden group hover:shadow-md transition-all cursor-pointer">
            <div className="absolute right-0 top-0 h-full w-1 bg-indigo-500"></div>
            <div className="flex justify-between items-start mb-4">
                <div className="p-2 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg text-indigo-600 dark:text-indigo-400"><span className="material-icons">center_focus_strong</span></div>
            </div>
            <div className="flex flex-col">
                <span className="text-2xl font-bold text-slate-900 dark:text-white mb-2 truncate">{mostActiveArea?.name || 'No Areas'}</span>
                <span className="text-sm font-medium text-slate-500 dark:text-slate-400">Most Active Area</span>
            </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Areas */}
        <div className="lg:col-span-2 space-y-6">
            <div className="flex items-center justify-between">
                <h2 className="text-xl font-bold text-slate-900 dark:text-white">Your Areas</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {data.areas.map(area => (
                    <div 
                        key={area.id}
                        onClick={() => onNavigate(`area:${area.id}`)}
                        className="group bg-white dark:bg-neutral-surface-dark rounded-xl p-5 shadow-sm border border-slate-200 dark:border-slate-700 hover:border-primary/50 transition-all cursor-pointer"
                    >
                        <div className="flex justify-between items-start mb-3">
                            <div className="flex items-center gap-3">
                                <span className={`w-3 h-3 rounded-full ${area.color} shadow-sm`}></span>
                                <h3 className="font-bold text-lg text-slate-900 dark:text-white">{area.name}</h3>
                            </div>
                        </div>
                        <p className="text-sm text-slate-500 mb-4 line-clamp-2">Tasks and notes related to {area.name}.</p>
                        <div className="flex items-center justify-between text-sm text-slate-500">
                             <div className="flex items-center gap-1"><span className="material-icons text-base">format_list_bulleted</span> {data.tasks.filter(t=>t.area_id === area.id && !t.completed).length} Pending</div>
                        </div>
                    </div>
                ))}
            </div>
        </div>


      </div>
    </div>
  );
};

export default Dashboard;